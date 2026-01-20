/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Approval manager for vuhlp stream-json protocol.
 * Coordinates between the input loop (receives approval.resolved) and
 * tool execution (waits for approval before proceeding).
 */

export interface ApprovalResolution {
  status: 'approved' | 'denied';
  modifiedArgs?: Record<string, unknown>;
}

interface PendingApproval {
  resolve: (resolution: ApprovalResolution) => void;
  reject: (error: Error) => void;
  toolName: string;
  toolId: string;
}

const pendingApprovals = new Map<string, PendingApproval>();

/**
 * Request approval for a tool call. Emits approval.requested event and
 * returns a promise that resolves when approval.resolved is received.
 */
export function requestApproval(
  approvalId: string,
  toolName: string,
  toolId: string,
  args: Record<string, unknown>,
): Promise<ApprovalResolution> {
  // Emit approval.requested event
  console.log(JSON.stringify({
    type: 'approval.requested',
    timestamp: new Date().toISOString(),
    approvalId,
    tool: {
      id: toolId,
      name: toolName,
      args,
    },
  }));

  return new Promise((resolve, reject) => {
    pendingApprovals.set(approvalId, {
      resolve,
      reject,
      toolName,
      toolId,
    });
  });
}

/**
 * Resolve a pending approval. Called when approval.resolved input is received.
 */
export function resolveApproval(
  approvalId: string,
  resolution: ApprovalResolution,
): boolean {
  const pending = pendingApprovals.get(approvalId);
  if (!pending) {
    return false;
  }

  pendingApprovals.delete(approvalId);
  pending.resolve(resolution);
  return true;
}

/**
 * Check if there are any pending approvals.
 */
export function hasPendingApprovals(): boolean {
  return pendingApprovals.size > 0;
}

/**
 * Cancel all pending approvals (e.g., on session end).
 */
export function cancelAllApprovals(): void {
  pendingApprovals.forEach((pending, id) => {
    pending.reject(new Error(`Approval ${id} cancelled: session ended`));
  });
  pendingApprovals.clear();
}

/**
 * Generate a unique approval ID.
 */
export function generateApprovalId(): string {
  return `approval-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}
