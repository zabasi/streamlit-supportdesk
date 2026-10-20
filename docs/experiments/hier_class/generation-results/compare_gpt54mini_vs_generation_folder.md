# Policy v3 gpt-5.4-mini vs generation-folder best vs grounded advanced

Generated 2026-10-16T15:42:09+00:00.

Same 15 annotated development cases. Gold labels: 6 answer / 6 request_info / 3 escalate.

- **Generation-folder best** is policy v3, `gpt-5.4-mini`, 81 curated gold passages (not retrieval).
- **This live run** is the same method and evidence, executed from this repo.
- **grounded advanced** is the best of the earlier grounded generators in this repo (`gpt-4o-mini` on retrieved packets). It is a harder setting.

## Costs (this live run)

Model `gpt-5.4-mini`. Evidence: gold passages.

| | USD |
|---|---:|
| 15 unique cases | **0.256039** |
| Actually billed (includes one duplicate DEV-10805 call) | 0.279674 |
| Generation-folder v3 (same 15, original) | 0.215597 |
| grounded advanced generation (`gpt-4o-mini`) | 0.013177 |

| case_id | outcome | input tokens | output tokens | cost USD |
|---|---|---:|---:|---:|
| DEV-5604 | escalate | 3156 | 3890 | 0.019872 |
| DEV-7044 | answer | 3040 | 3382 | 0.017499 |
| DEV-9342 | request_info | 3220 | 1877 | 0.0108615 |
| DEV-4338 | answer | 3655 | 5219 | 0.02622675 |
| DEV-6166 | escalate | 3221 | 2442 | 0.01340475 |
| DEV-16388 | escalate | 2663 | 2515 | 0.01331475 |
| DEV-3903 | answer | 2941 | 3242 | 0.01679475 |
| DEV-4325 | answer | 4279 | 4434 | 0.02316225 |
| DEV-7305 | answer | 2771 | 4249 | 0.02119875 |
| DEV-3585 | escalate | 2419 | 1531 | 0.00870375 |
| DEV-3842 | answer | 2458 | 2416 | 0.0127155 |
| DEV-4082 | answer | 3264 | 3886 | 0.019935 |
| DEV-5591 | answer | 2853 | 2665 | 0.01413225 |
| DEV-4989 | answer | 3885 | 3087 | 0.01680525 |
| DEV-10805 | answer | 4034 | 4086 | 0.0214125 |
| **TOTAL (15 unique)** |  |  |  | **0.256039** |

## Outcome comparison

| method | model | evidence | correct | accuracy | cost USD |
|---|---|---|---:|---:|---:|
| generation folder policy v3 (best of that project) | `gpt-5.4-mini` | 81 gold passages | 9/15 | 0.6000 | 0.215597 |
| this repo policy v3 live | `gpt-5.4-mini` | 81 gold passages | 9/15 | 0.6 | 0.256039 |
| this repo grounded checker + advanced retrieve | `gpt-4o-mini` | retrieval packet | 6/15 | 0.4 | 0.013177 |

### Generation folder (best of that project)

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 5 | 0 | 1 | 6 |
| request_info | 0 | 2 | 4 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 6 | 2 | 7 | 15 |

### This live policy v3

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 6 | 0 | 0 | 6 |
| request_info | 3 | 1 | 2 | 6 |
| escalate | 1 | 0 | 2 | 3 |
| **total** | 10 | 1 | 4 | 15 |

### grounded advanced

| gold \ predicted | answer | request_info | escalate | total |
|---|---:|---:|---:|---:|
| answer | 3 | 2 | 1 | 6 |
| request_info | 3 | 2 | 1 | 6 |
| escalate | 1 | 1 | 1 | 3 |
| **total** | 7 | 5 | 3 | 15 |

## Per-case actions

| case_id | gold | generation-folder v3 | this live v3 | grounded advanced | live cost | folder cost | grounded cost |
|---|---|---|---|---|---:|---:|---:|
| DEV-5604 | request_info | escalate | escalate | escalate | 0.019872 | 0.0123165 | 0.00087495 |
| DEV-7044 | request_info | request_info | answer | answer | 0.017499 | 0.0092415 | 0.0009342 |
| DEV-9342 | request_info | escalate | request_info | request_info | 0.0108615 | 0.009579 | 0.0007428 |
| DEV-4338 | answer | escalate | answer | escalate | 0.02622675 | 0.01447725 | 0.0008829 |
| DEV-6166 | escalate | escalate | escalate | escalate | 0.01340475 | 0.00745575 | 0.00068325 |
| DEV-16388 | escalate | answer | escalate | answer | 0.01331475 | 0.01533075 | 0.0010218 |
| DEV-3903 | request_info | escalate | answer | answer | 0.01679475 | 0.01902675 | 0.0009273 |
| DEV-4325 | answer | answer | answer | request_info | 0.02316225 | 0.01429725 | 0.0008523 |
| DEV-7305 | answer | answer | answer | answer | 0.02119875 | 0.01200525 | 0.0009465 |
| DEV-3585 | request_info | escalate | escalate | answer | 0.00870375 | 0.00815925 | 0.00087765 |
| DEV-3842 | answer | answer | answer | request_info | 0.0127155 | 0.014664 | 0.0007956 |
| DEV-4082 | answer | answer | answer | answer | 0.019935 | 0.0197685 | 0.0009537 |
| DEV-5591 | request_info | request_info | answer | request_info | 0.01413225 | 0.01495125 | 0.00074325 |
| DEV-4989 | answer | answer | answer | answer | 0.01680525 | 0.02481525 | 0.00110115 |
| DEV-10805 | escalate | escalate | answer | request_info | 0.0214125 | 0.019509 | 0.00084 |

## Which method is best

**Policy v3 with `gpt-5.4-mini` on gold passages is the best generation method here.**
Both the generation-folder run and this live rerun hit **9/15 (0.60)**. grounded `advanced`
on retrieved packets is **6/15 (0.40)**.

Tie-break between the two v3 samples: this live run is the better sample.

- Answerable: this live **6/6**, folder 5/6, the grounded checker 3/6.
- Escalation: this live **2/3** (it correctly escalated DEV-16388; the folder answered).
- Request_info: folder **2/6**, this live 1/6, the grounded checker 2/6. Clarification is still the weak cell.

This is **not** a RAG ranking. The 9/15 methods saw reviewer-selected passages. the grounded checker only
saw six retrieved chunks, so a lower score there does not mean the packet checker is
useless — it means retrieval-conditioned generation is harder.

Cost: this live 15-case unique total is **$0.256039**, vs $0.215597 for the
original folder run and $0.013177 for the grounded checker `gpt-4o-mini`. Do not treat
contract-valid JSON as a verified Streamlit patch.
