"""Integration template; importing or running this file makes no network calls."""
from supportdesk_generation import generate_from_packet,to_pipeline_answer

def generation_node(state,complete,*,questions_used=0,history=None):
    # complete is supplied by your pipeline. It receives the complete request
    # and must return a raw Chat Completions response dictionary.
    result=generate_from_packet(state['case'],state['packet'],complete,
        history=history,questions_used=questions_used)
    answer=to_pipeline_answer(result)
    from dataclasses import asdict
    return {'generated':asdict(answer),'answer':asdict(answer),
        'summary':result['maintainer_summary'],
        'extra':{**state.get('extra',{}),'controlled_generation':result}}
