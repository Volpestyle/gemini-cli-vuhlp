/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type {
  Config,
  ToolCallRequestInfo,
  ResumedSessionData,
  CompletedToolCall,
  UserFeedbackPayload,
} from '@google/gemini-cli-core';
import { isSlashCommand } from './ui/utils/commandUtils.js';
import type { LoadedSettings } from './config/settings.js';
import {
  executeToolCall,
  GeminiEventType,
  FatalInputError,
  promptIdContext,
  OutputFormat,
  JsonFormatter,
  StreamJsonFormatter,
  JsonStreamEventType,
  uiTelemetryService,
  debugLogger,
  coreEvents,
  CoreEvent,
  createWorkingStdio,
  recordToolCallInteractions,
  ToolErrorType,
} from '@google/gemini-cli-core';

import type { Content, Part } from '@google/genai';
import readline from 'node:readline';

import { convertSessionToHistoryFormats } from './ui/hooks/useSessionBrowser.js';
import { handleSlashCommand } from './nonInteractiveCliCommands.js';
import { ConsolePatcher } from './ui/utils/ConsolePatcher.js';
import { handleAtCommand } from './ui/hooks/atCommandProcessor.js';
import {
  handleError,
  handleToolError,
  handleCancellationError,
  handleMaxTurnsExceededError,
} from './utils/errors.js';
import { TextOutput } from './ui/utils/textOutput.js';
import {
  requestApproval,
  generateApprovalId,
} from './approvalManager.js';

interface RunNonInteractiveParams {
  config: Config;
  settings: LoadedSettings;
  input: string;
  prompt_id: string;
  resumedSessionData?: ResumedSessionData;
  /** If true, tool execution requires approval via stream-json protocol */
  requireToolApproval?: boolean;
}

export async function runNonInteractive({
  config,
  settings,
  input,
  prompt_id,
  resumedSessionData,
  requireToolApproval = false,
}: RunNonInteractiveParams): Promise<void> {
  return promptIdContext.run(prompt_id, async () => {
    const consolePatcher = new ConsolePatcher({
      stderr: true,
      debugMode: config.getDebugMode(),
      onNewMessage: (msg) => {
        coreEvents.emitConsoleLog(msg.type, msg.content);
      },
    });
    const { stdout: workingStdout } = createWorkingStdio();
    const textOutput = new TextOutput(workingStdout);

    const handleUserFeedback = (payload: UserFeedbackPayload) => {
      const prefix = payload.severity.toUpperCase();
      process.stderr.write(`[${prefix}] ${payload.message}\n`);
      if (payload.error && config.getDebugMode()) {
        const errorToLog =
          payload.error instanceof Error
            ? payload.error.stack || payload.error.message
            : String(payload.error);
        process.stderr.write(`${errorToLog}\n`);
      }
    };

    const startTime = Date.now();
    const streamFormatter =
      config.getOutputFormat() === OutputFormat.STREAM_JSON
        ? new StreamJsonFormatter()
        : null;

    const abortController = new AbortController();

    // Track cancellation state
    let isAborting = false;
    let cancelMessageTimer: NodeJS.Timeout | null = null;

    // Setup stdin listener for Ctrl+C detection
    let stdinWasRaw = false;
    let rl: readline.Interface | null = null;

    const setupStdinCancellation = () => {
      // Only setup if stdin is a TTY (user can interact)
      if (!process.stdin.isTTY) {
        return;
      }

      // Save original raw mode state
      stdinWasRaw = process.stdin.isRaw || false;

      // Enable raw mode to capture individual keypresses
      process.stdin.setRawMode(true);
      process.stdin.resume();

      // Setup readline to emit keypress events
      rl = readline.createInterface({
        input: process.stdin,
        escapeCodeTimeout: 0,
      });
      readline.emitKeypressEvents(process.stdin, rl);

      // Listen for Ctrl+C
      const keypressHandler = (
        str: string,
        key: { name?: string; ctrl?: boolean },
      ) => {
        // Detect Ctrl+C: either ctrl+c key combo or raw character code 3
        if ((key && key.ctrl && key.name === 'c') || str === '\u0003') {
          // Only handle once
          if (isAborting) {
            return;
          }

          isAborting = true;

          // Only show message if cancellation takes longer than 200ms
          // This reduces verbosity for fast cancellations
          cancelMessageTimer = setTimeout(() => {
            process.stderr.write('\nCancelling...\n');
          }, 200);

          abortController.abort();
          // Note: Don't exit here - let the abort flow through the system
          // and trigger handleCancellationError() which will exit with proper code
        }
      };

      process.stdin.on('keypress', keypressHandler);
    };

    const cleanupStdinCancellation = () => {
      // Clear any pending cancel message timer
      if (cancelMessageTimer) {
        clearTimeout(cancelMessageTimer);
        cancelMessageTimer = null;
      }

      // Cleanup readline and stdin listeners
      if (rl) {
        rl.close();
        rl = null;
      }

      // Remove keypress listener
      process.stdin.removeAllListeners('keypress');

      // Restore stdin to original state
      if (process.stdin.isTTY) {
        process.stdin.setRawMode(stdinWasRaw);
        process.stdin.pause();
      }
    };

    let errorToHandle: unknown | undefined;
    try {
      consolePatcher.patch();

      // Setup stdin cancellation listener
      setupStdinCancellation();

      coreEvents.on(CoreEvent.UserFeedback, handleUserFeedback);
      coreEvents.drainBacklogs();

      // Handle EPIPE errors when the output is piped to a command that closes early.
      process.stdout.on('error', (err: NodeJS.ErrnoException) => {
        if (err.code === 'EPIPE') {
          // Exit gracefully if the pipe is closed.
          process.exit(0);
        }
      });

      const geminiClient = config.getGeminiClient();

      // Initialize chat.  Resume if resume data is passed.
      if (resumedSessionData) {
        await geminiClient.resumeChat(
          convertSessionToHistoryFormats(
            resumedSessionData.conversation.messages,
          ).clientHistory,
          resumedSessionData,
        );
      }

      // Emit init event for streaming JSON
      if (streamFormatter) {
        streamFormatter.emitEvent({
          type: JsonStreamEventType.INIT,
          timestamp: new Date().toISOString(),
          session_id: config.getSessionId(),
          model: config.getModel(),
        });
      }

      let query: Part[] | undefined;

      if (isSlashCommand(input)) {
        const slashCommandResult = await handleSlashCommand(
          input,
          abortController,
          config,
          settings,
        );
        // If a slash command is found and returns a prompt, use it.
        // Otherwise, slashCommandResult falls through to the default prompt
        // handling.
        if (slashCommandResult) {
          query = slashCommandResult as Part[];
        }
      }

      if (!query) {
        const { processedQuery, error } = await handleAtCommand({
          query: input,
          config,
          addItem: (_item, _timestamp) => 0,
          onDebugMessage: () => { },
          messageId: Date.now(),
          signal: abortController.signal,
        });

        if (error || !processedQuery) {
          // An error occurred during @include processing (e.g., file not found).
          // The error message is already logged by handleAtCommand.
          throw new FatalInputError(
            error || 'Exiting due to an error processing the @ command.',
          );
        }
        query = processedQuery as Part[];
      }

      // Emit user message event for streaming JSON
      if (streamFormatter) {
        streamFormatter.emitEvent({
          type: JsonStreamEventType.MESSAGE,
          timestamp: new Date().toISOString(),
          role: 'user',
          content: input,
        });
      }

      let currentMessages: Content[] = [{ role: 'user', parts: query }];

      let turnCount = 0;
      while (true) {
        turnCount++;
        if (
          config.getMaxSessionTurns() >= 0 &&
          turnCount > config.getMaxSessionTurns()
        ) {
          handleMaxTurnsExceededError(config);
        }
        const toolCallRequests: ToolCallRequestInfo[] = [];

        const responseStream = geminiClient.sendMessageStream(
          currentMessages[0]?.parts || [],
          abortController.signal,
          prompt_id,
        );

        let responseText = '';
        for await (const event of responseStream) {
          if (abortController.signal.aborted) {
            handleCancellationError(config);
          }

          if (event.type === GeminiEventType.Content) {
            if (streamFormatter) {
              streamFormatter.emitEvent({
                type: JsonStreamEventType.MESSAGE,
                timestamp: new Date().toISOString(),
                role: 'assistant',
                content: event.value,
                delta: true,
              });
            } else if (config.getOutputFormat() === OutputFormat.JSON) {
              responseText += event.value;
            } else {
              if (event.value) {
                textOutput.write(event.value);
              }
            }
          } else if (event.type === GeminiEventType.ToolCallRequest) {
            if (streamFormatter) {
              streamFormatter.emitEvent({
                type: JsonStreamEventType.TOOL_USE,
                timestamp: new Date().toISOString(),
                tool_name: event.value.name,
                tool_id: event.value.callId,
                parameters: event.value.args,
              });
            }
            toolCallRequests.push(event.value);
          } else if (event.type === GeminiEventType.LoopDetected) {
            if (streamFormatter) {
              streamFormatter.emitEvent({
                type: JsonStreamEventType.ERROR,
                timestamp: new Date().toISOString(),
                severity: 'warning',
                message: 'Loop detected, stopping execution',
              });
            }
          } else if (event.type === GeminiEventType.MaxSessionTurns) {
            if (streamFormatter) {
              streamFormatter.emitEvent({
                type: JsonStreamEventType.ERROR,
                timestamp: new Date().toISOString(),
                severity: 'error',
                message: 'Maximum session turns exceeded',
              });
            }
          } else if (event.type === GeminiEventType.Error) {
            throw event.value.error;
          } else if (event.type === GeminiEventType.AgentExecutionStopped) {
            const stopMessage = `Agent execution stopped: ${event.value.systemMessage?.trim() || event.value.reason}`;
            if (config.getOutputFormat() === OutputFormat.TEXT) {
              process.stderr.write(`${stopMessage}\n`);
            }
            // Emit final result event for streaming JSON if needed
            if (streamFormatter) {
              const metrics = uiTelemetryService.getMetrics();
              const durationMs = Date.now() - startTime;

              // Emit telemetry.usage for vuhlp protocol compatibility
              const modelName = config.getModel();
              const modelMetrics = metrics.models[modelName];
              const inputTokens = modelMetrics?.tokens?.input ?? 0;
              const outputTokens = modelMetrics?.tokens?.candidates ?? 0;
              console.log(JSON.stringify({
                type: 'telemetry.usage',
                timestamp: new Date().toISOString(),
                provider: 'gemini',
                model: modelName,
                usage: {
                  promptTokens: inputTokens,
                  completionTokens: outputTokens,
                  totalTokens: inputTokens + outputTokens,
                },
              }));

              streamFormatter.emitEvent({
                type: JsonStreamEventType.RESULT,
                timestamp: new Date().toISOString(),
                status: 'success',
                stats: streamFormatter.convertToStreamStats(
                  metrics,
                  durationMs,
                ),
              });
            }
            return;
          } else if (event.type === GeminiEventType.AgentExecutionBlocked) {
            const blockMessage = `Agent execution blocked: ${event.value.systemMessage?.trim() || event.value.reason}`;
            if (config.getOutputFormat() === OutputFormat.TEXT) {
              process.stderr.write(`[WARNING] ${blockMessage}\n`);
            }
          } else if (event.type === GeminiEventType.ContextWindowWillOverflow) {
            const { estimatedRequestTokenCount, remainingTokenCount } =
              event.value;
            const msg = `Context window overflow: Estimated request tokens (${estimatedRequestTokenCount}) exceeds remaining tokens (${remainingTokenCount}).`;
            if (streamFormatter) {
              streamFormatter.emitEvent({
                type: JsonStreamEventType.ERROR,
                timestamp: new Date().toISOString(),
                severity: 'error',
                message: msg,
              });
            }
            throw new FatalInputError(msg);
          }
        }

        if (toolCallRequests.length > 0) {
          textOutput.ensureTrailingNewline();
          const toolResponseParts: Part[] = [];
          const completedToolCalls: CompletedToolCall[] = [];
          const requiresApproval = streamFormatter && requireToolApproval;

          for (const requestInfo of toolCallRequests) {
            // Request approval if in stream-json mode with approval enabled
            if (requiresApproval) {
              const approvalId = generateApprovalId();
              const resolution = await requestApproval(
                approvalId,
                requestInfo.name,
                requestInfo.callId,
                requestInfo.args as Record<string, unknown>,
              );

              // Emit approval.resolved event for vuhlp protocol
              console.log(JSON.stringify({
                type: 'approval.resolved',
                timestamp: new Date().toISOString(),
                approvalId,
                resolution,
              }));

              if (resolution.status === 'denied') {
                // Skip this tool call if denied
                if (streamFormatter) {
                  streamFormatter.emitEvent({
                    type: JsonStreamEventType.TOOL_RESULT,
                    timestamp: new Date().toISOString(),
                    tool_id: requestInfo.callId,
                    status: 'error',
                    error: {
                      type: 'APPROVAL_DENIED',
                      message: `Tool execution denied by user`,
                    },
                  });
                }
                continue;
              }

              // Apply modified args if provided
              if (resolution.modifiedArgs) {
                Object.assign(requestInfo.args, resolution.modifiedArgs);
              }
            }

            const completedToolCall = await executeToolCall(
              config,
              requestInfo,
              abortController.signal,
            );
            const toolResponse = completedToolCall.response;

            completedToolCalls.push(completedToolCall);

            if (streamFormatter) {
              streamFormatter.emitEvent({
                type: JsonStreamEventType.TOOL_RESULT,
                timestamp: new Date().toISOString(),
                tool_id: requestInfo.callId,
                status: toolResponse.error ? 'error' : 'success',
                output:
                  typeof toolResponse.resultDisplay === 'string'
                    ? toolResponse.resultDisplay
                    : undefined,
                error: toolResponse.error
                  ? {
                    type: toolResponse.errorType || 'TOOL_EXECUTION_ERROR',
                    message: toolResponse.error.message,
                  }
                  : undefined,
              });
            }

            if (toolResponse.error) {
              handleToolError(
                requestInfo.name,
                toolResponse.error,
                config,
                toolResponse.errorType || 'TOOL_EXECUTION_ERROR',
                typeof toolResponse.resultDisplay === 'string'
                  ? toolResponse.resultDisplay
                  : undefined,
              );
            }

            if (toolResponse.responseParts) {
              toolResponseParts.push(...toolResponse.responseParts);
            }
          }

          // Record tool calls with full metadata before sending responses to Gemini
          try {
            const currentModel =
              geminiClient.getCurrentSequenceModel() ?? config.getModel();
            geminiClient
              .getChat()
              .recordCompletedToolCalls(currentModel, completedToolCalls);

            await recordToolCallInteractions(config, completedToolCalls);
          } catch (error) {
            debugLogger.error(
              `Error recording completed tool call information: ${error}`,
            );
          }

          // Check if any tool requested to stop execution immediately
          const stopExecutionTool = completedToolCalls.find(
            (tc) => tc.response.errorType === ToolErrorType.STOP_EXECUTION,
          );

          if (stopExecutionTool && stopExecutionTool.response.error) {
            const stopMessage = `Agent execution stopped: ${stopExecutionTool.response.error.message}`;

            if (config.getOutputFormat() === OutputFormat.TEXT) {
              process.stderr.write(`${stopMessage}\n`);
            }

            // Emit final result event for streaming JSON
            if (streamFormatter) {
              const metrics = uiTelemetryService.getMetrics();
              const durationMs = Date.now() - startTime;

              // Emit telemetry.usage for vuhlp protocol compatibility
              const modelName2 = config.getModel();
              const modelMetrics2 = metrics.models[modelName2];
              const inputTokens2 = modelMetrics2?.tokens?.input ?? 0;
              const outputTokens2 = modelMetrics2?.tokens?.candidates ?? 0;
              console.log(JSON.stringify({
                type: 'telemetry.usage',
                timestamp: new Date().toISOString(),
                provider: 'gemini',
                model: modelName2,
                usage: {
                  promptTokens: inputTokens2,
                  completionTokens: outputTokens2,
                  totalTokens: inputTokens2 + outputTokens2,
                },
              }));

              streamFormatter.emitEvent({
                type: JsonStreamEventType.RESULT,
                timestamp: new Date().toISOString(),
                status: 'success',
                stats: streamFormatter.convertToStreamStats(
                  metrics,
                  durationMs,
                ),
              });
            } else if (config.getOutputFormat() === OutputFormat.JSON) {
              const formatter = new JsonFormatter();
              const stats = uiTelemetryService.getMetrics();
              textOutput.write(
                formatter.format(config.getSessionId(), responseText, stats),
              );
            } else {
              textOutput.ensureTrailingNewline(); // Ensure a final newline
            }
            return;
          }

          currentMessages = [{ role: 'user', parts: toolResponseParts }];
        } else {
          // Emit final result event for streaming JSON
          if (streamFormatter) {
            const metrics = uiTelemetryService.getMetrics();
            const durationMs = Date.now() - startTime;

            // Emit telemetry.usage for vuhlp protocol compatibility
            const modelName3 = config.getModel();
            const modelMetrics3 = metrics.models[modelName3];
            const inputTokens3 = modelMetrics3?.tokens?.input ?? 0;
            const outputTokens3 = modelMetrics3?.tokens?.candidates ?? 0;
            console.log(JSON.stringify({
              type: 'telemetry.usage',
              timestamp: new Date().toISOString(),
              provider: 'gemini',
              model: modelName3,
              usage: {
                promptTokens: inputTokens3,
                completionTokens: outputTokens3,
                totalTokens: inputTokens3 + outputTokens3,
              },
            }));

            streamFormatter.emitEvent({
              type: JsonStreamEventType.RESULT,
              timestamp: new Date().toISOString(),
              status: 'success',
              stats: streamFormatter.convertToStreamStats(metrics, durationMs),
            });
          } else if (config.getOutputFormat() === OutputFormat.JSON) {
            const formatter = new JsonFormatter();
            const stats = uiTelemetryService.getMetrics();
            textOutput.write(
              formatter.format(config.getSessionId(), responseText, stats),
            );
          } else {
            textOutput.ensureTrailingNewline(); // Ensure a final newline
          }
          return;
        }
      }
    } catch (error) {
      errorToHandle = error;
    } finally {
      // Cleanup stdin cancellation before other cleanup
      cleanupStdinCancellation();

      consolePatcher.cleanup();
      coreEvents.off(CoreEvent.UserFeedback, handleUserFeedback);
    }

    if (errorToHandle) {
      handleError(errorToHandle, config);
    }
  });
}
