import chainlit as cl
from time import perf_counter

from rag_workflow import RAGWorkflow, UIProgressEvent
from logger_setup import setup_logger, session_context

logger, log_listener = setup_logger()

@cl.on_chat_start
async def on_chat_start():
    
    session_id = cl.user_session.get("id") or "anonymous_session"

    workflow = RAGWorkflow(session_id=session_id, timeout=120)

    cl.user_session.set("workflow", workflow)
    cl.user_session.set("session_id", session_id)
    
    logger.info("New Chainlit session started.", extra={"chainlit_session": session_id})

@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    workflow: RAGWorkflow = cl.user_session.get("workflow")
    if not workflow:
        await cl.Message(content="Session error. Please refresh the page.", author="Ariadne AI Assistant").send()
        return
    
    session_context.set(session_id)

    try:
        start_time = perf_counter()
        logger.info("UI: Received user message", extra={"user_msg": message.content})


        handler = workflow.run(user_msg=message.content)

        async with cl.Step(name='Agent Reasoning', type='run') as parent_step:
            async for event in handler.stream_events():
                if isinstance(event, UIProgressEvent):
                    async with cl.Step(name=event.step_name, type='tool', parent_id=parent_step.id) as child_step:
                        child_step.output = event.msg
                        await child_step.send()

            parent_step.name='Ολοκληρώθηκε η απάντηση'      
            await parent_step.update()
               
        response_stream = await handler
        
        msg = cl.Message(content="", author='Ariadne AI Assistant')
        await msg.send()

        first_token_flag = False
        async for token in response_stream:
            if not first_token_flag:
                ttft = perf_counter() - start_time
                logger.info("First token generated", extra={"ttft_seconds": round(ttft, 3)})
                print(f"⏱️ TTFT: {ttft:.3f} seconds")
                first_token_flag = True

            await msg.stream_token(token.delta or "")
        
        total_time = perf_counter() - start_time
        logger.info("Response completed", extra={"total_time_seconds": round(total_time, 3)})
        msg.content += f"\n\n*(Χρόνος απόκρισης: {total_time:.2f} δευτερόλεπτα)*"
        await msg.update()

    except Exception as e:
        logger.error(f"Workflow failed during message processing: {str(e)}", exc_info=True)

        await cl.Message(
            content="Προέκυψε ένα σφάλμα κατά την επεξεργασία. Παρακαλώ δοκιμάστε ξανά.", 
            author='Ariadne AI Assistant'
        ).send()

@cl.on_chat_end
def on_chat_end():
    session_id = cl.user_session.get("session_id")
    session_context.set(session_id) # ensure log knows who is disconnecting
    workflow = cl.user_session.get("workflow")

    if workflow:
        del workflow
    cl.user_session.set("workflow", None)

    logger.info("Session disconnected. Resources cleared.")
    