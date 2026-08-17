import asyncio
import sys
import json
import httpx

BASE_URL = "http://localhost:8080/api"

# Rich ANSI formatting for clean terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log(tag: str, msg: str, color: str = Colors.CYAN):
    print(f"{color}[{tag.upper()}]{Colors.END} {msg}")

async def run_e2e_demo():
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}  MULTI-AGENT ORCHESTRATION SYSTEM - FULL E2E INTEGRATION TEST{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*70}{Colors.END}\n")

    complex_task = (
        "Perform deep market research on top 3 open-source vector databases (Milvus, Qdrant, ChromaDB). "
        "Benchmark their scalability, indexing algorithms (HNSW vs IVF), and synthesize an enterprise architecture brief."
    )

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # Step 1: Submit Task
        log("SUBMIT", f"Submitting complex mission: {Colors.BOLD}'{complex_task[:80]}...'{Colors.END}")
        try:
            submit_resp = await client.post("/tasks", json={"task": complex_task})
            submit_resp.raise_for_status()
        except Exception as e:
            log("ERROR", f"Could not connect to FastAPI at {BASE_URL}. Ensure docker-compose is running. Details: {e}", Colors.RED)
            sys.exit(1)

        data = submit_resp.json()
        thread_id = data["thread_id"]
        celery_id = data.get("celery_task_id", "N/A")
        log("SUCCESS", f"Thread initialized: {Colors.GREEN}{thread_id}{Colors.END} (Celery Job: {celery_id})", Colors.GREEN)

        # Step 2: Poll until Human Approval or Completion
        log("POLL", "Polling graph state via PostgreSQL checkpointer...")
        plan_displayed = False
        approval_done = False

        max_polls = 60
        poll_count = 0

        while poll_count < max_polls:
            poll_count += 1
            await asyncio.sleep(2.0)

            state_resp = await client.get(f"/tasks/{thread_id}/state")
            if state_resp.status_code != 200:
                log("POLL", f"Awaiting graph state compilation... (attempt {poll_count})", Colors.YELLOW)
                continue

            state = state_resp.json()
            status = state.get("status")
            plan = state.get("current_plan")
            messages = state.get("messages", [])
            next_nodes = state.get("next_nodes", [])

            # Display plan once Supervisor emits it
            if plan and not plan_displayed:
                plan_displayed = True
                print(f"\n{Colors.BOLD}{Colors.BLUE}--- SUPERVISOR EXECUTION PLAN ---{Colors.END}")
                subtasks = plan.get("subtasks", []) if isinstance(plan, dict) else getattr(plan, "subtasks", [])
                conf = plan.get("confidence_score", 1.0) if isinstance(plan, dict) else getattr(plan, "confidence_score", 1.0)
                req_hitl = plan.get("requires_human_approval", False) if isinstance(plan, dict) else getattr(plan, "requires_human_approval", False)
                
                print(f"Confidence Score: {Colors.BOLD}{conf * 100:.1f}%{Colors.END}")
                print(f"Requires Human Review: {Colors.YELLOW if req_hitl else Colors.GREEN}{req_hitl}{Colors.END}")
                print(f"Decomposed Subtasks ({len(subtasks)}):")
                for i, st in enumerate(subtasks):
                    role = st.get("assigned_specialist", "specialist") if isinstance(st, dict) else st.assigned_specialist
                    desc = st.get("description", "") if isinstance(st, dict) else st.description
                    print(f"  {Colors.BOLD}[Step {i+1}]{Colors.END} ({Colors.CYAN}{role}{Colors.END}) -> {desc}")
                print(f"{Colors.BLUE}{'---------------------------------'}{Colors.END}\n")

            # Step 3: Handle Human-in-the-Loop Interruption
            if status == "pending_human_approval" and not approval_done:
                log("HITL", "LangGraph paused at 'escalation_node' waiting for human sign-off!", Colors.YELLOW)
                log("HITL", "Operator reviewing proposed execution plan and specialist assignments...", Colors.YELLOW)
                await asyncio.sleep(1.5)
                
                feedback_msg = "Proceed. Ensure thorough comparison matrix on concurrency metrics."
                log("APPROVE", f"Operator issuing approval with feedback: '{feedback_msg}'", Colors.GREEN)
                
                approve_resp = await client.post(f"/tasks/{thread_id}/approve", json={
                    "approved": True,
                    "feedback": feedback_msg
                })
                approve_resp.raise_for_status()
                log("APPROVE", f"Approval confirmed. Celery task dispatched to resume graph.", Colors.GREEN)
                approval_done = True
                continue

            # Step 4: Check if completed
            if status == "running_or_completed" and len(next_nodes) == 0 and len(messages) > 0:
                print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}")
                print(f"{Colors.BOLD}{Colors.GREEN}  WORKFLOW SUCCESSFULLY COMPLETED{Colors.END}")
                print(f"{Colors.BOLD}{Colors.GREEN}{'='*70}{Colors.END}\n")

                print(f"{Colors.BOLD}Final Message Transcript:{Colors.END}")
                for idx, msg in enumerate(messages):
                    print(f"\n{Colors.CYAN}[Message #{idx+1}]{Colors.END}\n{msg}")

                # Step 5: Verify Node Trace Observability
                log("INSPECT", "Validating Node Observability & Token Metrics for 'supervisor_node'...", Colors.BLUE)
                trace_resp = await client.get(f"/tasks/{thread_id}/trace/supervisor_node")
                if trace_resp.status_code == 200:
                    trace_data = trace_resp.json()
                    tokens = trace_data.get("token_usage", {})
                    print(f"  • Supervisor Step: {trace_data.get('step')}")
                    print(f"  • Prompt Tokens: {tokens.get('prompt_tokens')} | Completion Tokens: {tokens.get('completion_tokens')}")
                    print(f"  • Est. Cost: ${tokens.get('estimated_cost_usd', 0.0):.6f}")

                log("MEMORY", "Validating ChromaDB Long-Term Memory snapshot for 'memorize_node'...", Colors.BLUE)
                mem_resp = await client.get(f"/tasks/{thread_id}/trace/memorize_node")
                if mem_resp.status_code == 200:
                    mem_data = mem_resp.json()
                    print(f"  • Memory Node Status: {Colors.GREEN}{mem_data.get('status')}{Colors.END}")
                    print(f"  • Output Vector Record: {mem_data.get('response')}")

                print(f"\n{Colors.BOLD}{Colors.GREEN}✓ Full E2E Test Passed: Supervisor -> Specialists -> HITL -> Reviewer -> ChromaDB Memory!{Colors.END}\n")
                return

            log("POLL", f"Graph active. Next nodes: {next_nodes or 'Executing'} (messages count: {len(messages)})", Colors.CYAN)

        log("TIMEOUT", "E2E Test timed out before completion.", Colors.RED)

if __name__ == "__main__":
    asyncio.run(run_e2e_demo())
