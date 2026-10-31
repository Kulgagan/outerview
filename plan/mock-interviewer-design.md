# Local Voice Mock Interviewer: Design Doc

**Status:** draft v1 | **Audience:** me | **Language:** Python | **Runs:** 100% local, started from the terminal

A personal tool that plays a technical interviewer. It listens to me, watches my LeetCode screen, talks back with a local voice, and at the end scores me like a real hiring loop would. No cloud, no database, no deployment.

---

## 0. TL;DR

| Layer | Pick | Why | Fallback |
|---|---|---|---|
| Mic and turn detection | `sounddevice` + Silero VAD | Small, accurate, runs on CPU | WebRTC VAD |
| Speech to text | `faster-whisper` (`small.en`) | Simple, accurate, supports a vocabulary prompt for jargon | Moonshine streaming, Parakeet |
| Brain and eyes | Qwen3.5-4B (Q4_K_M GGUF) served by llama.cpp `llama-server` | One small model does conversation and screen reading, OpenAI-compatible API | Qwen3.5-2B if too slow; Qwen3.5-9B for the final scoring pass only |
| Voice | Kokoro-82M in-process (default), VoiceStudio over HTTP (optional quality upgrade) | Kokoro is the fast path; VoiceStudio is a studio app, not a low-latency engine | KittenTTS, Piper |
| Screen | `mss` + Pillow, background observer thread | Cheap capture, vision runs off the critical path | OCR on the editor crop |
| State | In-memory dataclasses | One interview, no history, no DB | none needed |
| Report | `rich` in the terminal | Stats for this interview only | optional `--save-report` markdown file |

Five decisions that shape everything else:

1. **Seeing is decoupled from talking.** A background observer keeps a cached snapshot of the problem, my code, and the last run result. The conversation never waits on a screenshot.
2. **The interviewer is mostly quiet.** Real interviewers talk far less than a chatbot. A rule engine decides when to speak, and the LLM decides what to say, starting every reply with a control tag like `[PASS]` or `[PROBE]`.
3. **Latency is managed, not hoped for.** Stream LLM output sentence by sentence into TTS, force 1 to 2 sentence replies, and play pre-rendered backchannels ("mm-hm") while the real answer is still being computed.
4. **Scoring is checklist-based.** Deterministic metrics (timings, hint count, run results) plus yes/no rubric questions the model must back with a transcript quote. Small models judge yes/no with evidence far better than they judge "rate this 1 to 4".
5. **TTS sits behind a tiny interface** so I can swap Kokoro for VoiceStudio (or anything else) with one config line.

---

## 1. Goals, non-goals, assumptions

### Goals

- Practice a full mock coding interview alone, by voice, on LeetCode problems.
- Feel conversational: the interviewer answers within about 1 to 1.5 seconds and sounds decent.
- Behave like a real interviewer: asks clarifying-question prompts, nudges me toward brute force first, asks for complexity, interrupts when I go silent or jump ahead, gives graded hints, asks follow-ups.
- Produce a per-interview scorecard with a hire / no-hire verdict based on how big-tech rubrics actually grade candidates.

### Non-goals

- No web UI, no accounts, no deployment, no multi-user support.
- No database and no history across interviews. Every run starts fresh and the stats die with the process (unless I pass `--save-report`).
- No browser automation or scraping of LeetCode. The tool only looks at my screen like a person would.
- No full-duplex speech-to-speech model. Plain half-duplex turn taking with barge-in is enough.

### Assumptions to confirm before building

| Item | Why it matters | Default if unknown |
|---|---|---|
| GPU (NVIDIA, how much VRAM) or Apple Silicon or CPU only | Decides model sizes and whether VoiceStudio is usable live | Plan for Profile A/B/C in section 4.3 and let Milestone 0 pick |
| OS | Screen and mic permissions (macOS needs both granted to the terminal app) | Cross-platform libraries only |
| Headphones | Without them the interviewer's voice leaks into the mic and triggers false barge-ins | Required |
| Which "VoiceStudio" | At least two GitHub projects use the name: `debpalash/VoiceStudio` and `msrbuilds/voice-studio` ("Voice Studio by MSR") | Design targets an OpenAI-compatible `/v1/audio/speech` endpoint, which the debpalash project documents |
| LeetCode layout | The observer crops the editor pane | Problem on the left, editor on the right, one monitor |

---

## 2. What the research says (and how it shapes the design)

### 2.1 How candidates are actually graded

Big-tech coding rubrics are surprisingly similar. The Tech Interview Handbook collates them into four dimensions, each scored on a 4-band scale (Strong hire, Leaning hire, Leaning no hire, Strong no hire). Google's published breakdown is Algorithms, Coding, Communication, Problem Solving (each 1 to 4). Meta's is Problem Solving, Coding, Verification, Communication. Different labels, same ground.

| Dimension | What the interviewer is looking for |
|---|---|
| **Communication** | Asks clarifying questions, explains approach and trade-offs, keeps talking while coding, organized and succinct |
| **Problem solving** | Understands the problem fast, systematic approach, reaches an optimized solution, states time and space complexity correctly, needs no major hints; advanced: multiple solutions with trade-offs, time left for follow-ups |
| **Technical competency (coding)** | Turns the plan into working code with few or no bugs, clean and readable, sensible use of language constructs |
| **Testing (verification)** | Tests typical cases and corner cases, finds and fixes own bugs, verifies systematically (for example stepping through like a debugger) |

The final call is holistic across dimensions, not a mechanical cut-off. Mixed signals across rounds get discussed in a debrief, which is why consistency matters.

### 2.2 What a hireable candidate does

- Clarifies inputs, constraints, and edge cases before touching the keyboard (but does not ask fifteen questions in a row, and does not ask things the statement already answers).
- States a brute force solution first, with its complexity, then optimizes. Brute force is a checkpoint that lets the interviewer help; it is not a weakness.
- Agrees on the approach with the interviewer before coding.
- Narrates while coding. The interviewer cannot give credit for thoughts they cannot hear.
- Takes hints gracefully: restates them, builds on them, does not ask for the answer directly.
- Tests with their own examples and edge cases, and walks through the code line by line.
- States final time and space complexity, correctly.
- Keeps pace. Google's guidance is to finish the first question in about 20 of 45 minutes to leave room for a follow-up.
- When stuck: says so out loud, proposes something imperfect, asks a targeted question. Does not go silent.

### 2.3 What gets a strong no-hire

- Staying silent, or being unable to explain the thought process even when asked.
- Jumping straight to code with no explanation.
- Announcing "done" without testing, or missing glaring bugs.
- Never reaching a working solution, with a disorganized approach.
- Leaning on repeated hints, or ignoring hints.

### 2.4 How a real interviewer behaves

Most of the research describes the candidate side, so this part is partly my synthesis of those sources. A typical 45 minute round:

| Time | Phase | Interviewer behavior |
|---|---|---|
| 0 to 2 | Intro and problem | States the problem briefly, invites questions, does not volunteer constraints |
| 2 to 7 | Clarify | Answers clarifying questions directly and briefly; asks for an example in my own words |
| 7 to 12 | Approach | "Let's start simple." Asks complexity of the brute force, then "can we do better?"; will say "go ahead and code" once the plan is sound |
| 12 to 30 | Coding | Mostly silent. Light backchannels. Chimes in after long silence ("talk me through it"), when I contradict myself, or when code diverges from the stated plan |
| 30 to 38 | Test and debug | "How would you test this?" Points at bugs by asking me to run an example, not by announcing them |
| 38 to 43 | Wrap and follow-up | "What is the complexity?" One follow-up (scale, new constraint, memory limit) |
| 43 to 45 | Close | Candidate questions |

Default session length for this tool: **35 minutes** (single problem plus one follow-up), configurable.

### 2.5 Design consequences

- The interviewer must be **terse and spoken-style**: 1 to 2 sentences, no lists, no markdown, no reading code aloud.
- It should **ask instead of assert** when correcting me. A 4B model will sometimes be wrong about a solution, and a question ("what happens on an empty array?") is safe even when the model is unsure.
- It needs **graded hints** (nudge, direction, pattern, near-solution) and must log each one, because hint usage is a core scoring input.
- Scoring needs both **behavioral timing data** (when did I first code, how long was I silent) and **content judgments** (did I state brute force, were the edge cases real).
- Bar calibration: default profile is **internship level**, where one hint at the lowest levels should not cap my problem-solving score. Profiles `intern`, `new_grad`, `senior` just change the hint penalties and the verdict thresholds in `scoring.py`.

---

## 3. Architecture

```
 mic --> VAD + endpointing --> STT --+
                                     |
 screen --> observer (diff + VLM) ---+--> EVENT QUEUE --> BRAIN --> sentence splitter --> TTS --> headphones
                                     |                      |   ^
 timers (silence, clock, phases) ----+                      |   | barge-in (stops audio and cancels LLM stream)
                                                            v
                                                 SESSION (in-memory log)
                                                            |
                                                            v
                                          scoring --> terminal report
```

### 3.1 Processes and threads

| Unit | Kind | Job |
|---|---|---|
| `llama-server` | Separate process | Hosts Qwen3.5-4B (text and vision), OpenAI-compatible HTTP API |
| VoiceStudio (optional) | Separate process or container | Only if I choose it as the TTS backend |
| `AudioIn` | Thread | Mic stream, VAD, cuts speech into utterances, raises barge-in |
| `STTWorker` | Thread | Utterance audio to text, pushes `UserTurn` events |
| `Observer` | Thread | Watches the screen, calls the VLM when the screen changed and settled |
| `Clock` | Thread | Emits trigger events (silence, time checks) once per second |
| `Brain` | Main thread | Consumes events, runs the phase machine, calls the LLM, streams sentences to TTS |
| `Speaker` | Thread | Synthesizes and plays audio, stoppable instantly |

Plain threads and `queue.Queue` are enough. No asyncio needed for a single-user tool.

### 3.2 Voice commands

Always listening for these (matched with a regex on the transcript, no LLM involved):

| Phrase | Effect |
|---|---|
| "start" | Begin the interview (works from idle only) |
| "repeat the question" | Interviewer restates the problem |
| "end interview" or "stop" | Jump to wrap-up and produce the report |
| "quit" | Exit the program (idle only) |

Fallback if STT mishears: pressing Enter in the terminal acts as "start".

---

## 4. Latency: the part that decides if this feels good

### 4.1 Budget (targets to verify in Milestone 0, not measurements)

| Stage | Target | Notes |
|---|---|---|
| End-of-speech detection | 500 to 700 ms | The single biggest cost. Shorter windows cut me off mid-thought |
| Final transcript | 100 to 250 ms | Short utterances; faster if partial transcription runs while I talk |
| LLM first sentence | 250 to 500 ms | Cached prompt prefix, tag plus 8 to 15 tokens |
| TTS first audio | 150 to 300 ms | Kokoro first chunk |
| **Total** | **about 1.0 to 1.7 s** | Reads as a human "thinking" pause |

Other people's fully local pipelines (Silero VAD, Parakeet, a small Qwen, local TTS) report roughly 0.7 to 1.5 seconds from end of speech to first audio, so this target is realistic but not free.

### 4.2 Techniques

1. **Stream everything.** LLM tokens go into a sentence splitter; each sentence goes straight to TTS while the next one is still generating.
2. **Short replies.** `max_tokens` around 90 and a prompt that demands 1 to 2 sentences. This is both realistic and the cheapest latency win.
3. **Backchannel clips.** At startup, pre-render a pool of short clips ("Mm-hm.", "Okay.", "Right.", "Go ahead.", "Take your time.") with the active voice. When I finish a thought, play one about 300 ms later while the real reply computes. Zero synthesis latency, and it matches how interviewers actually respond.
4. **Stable prompt prefix.** Persona, problem, and interviewer notes come first and never change; volatile state (phase, timer, code snapshot) goes last, so llama.cpp reuses its prompt cache.
5. **Thinking mode off** for every conversational call (see 5.3).
6. **Keep everything warm.** Load all models at startup and run one dummy inference through each.
7. **Vocabulary prompt for STT.** Seed Whisper with "hash map, two pointers, sliding window, time complexity, O(n), binary search, DFS, BFS, heap" so jargon is not mangled.

### 4.3 Hardware profiles

| Profile | Machine | LLM | STT | TTS | Notes |
|---|---|---|---|---|---|
| A | NVIDIA GPU, 8 GB or more | Qwen3.5-4B Q4_K_M on GPU (roughly 5.5 GB at 4-bit per Unsloth's sizing table) | `small.en` on CPU int8 or GPU if there is room | Kokoro on CPU or GPU | 8 GB is tight with everything on GPU; put STT and TTS on CPU first |
| B | Apple Silicon, 16 GB or more | Qwen3.5-4B via llama.cpp (Metal) | faster-whisper CPU, or an MLX Whisper or Parakeet build | Kokoro CPU (its README reports first-chunk latency under 1 s on an M3 Pro) | Grant screen recording and mic to the terminal app |
| C | CPU only | Qwen3.5-2B for talking, 4B only for the one-time screen read | Moonshine small streaming | Kokoro CPU (first chunk was about 3.5 s on an older i7 in its README) or KittenTTS | Expect slower, tune reply length down |

Milestone 0 measures each component alone and picks the profile.

---

## 5. Components

### 5.1 Audio in, VAD, endpointing

- 16 kHz mono, 32 ms frames from `sounddevice`.
- Silero VAD per frame. Start of speech when probability stays above about 0.5 for 250 ms; keep a 300 ms pre-roll so the first word is not clipped.
- **End-of-turn silence is phase dependent**:

| Phase | Silence that ends my turn | Reason |
|---|---|---|
| Idle, clarify, approach, follow-up | about 700 ms | Conversation, respond quickly |
| Coding, testing | about 1500 ms | I mutter and think aloud; do not cut in on every pause |

- **Barge-in:** if speech (at least 300 ms) starts while the interviewer is talking, stop playback, flush the TTS queue, and cancel the in-flight LLM stream. Requires headphones.
- Discard utterances under about 3 words unless they match a voice command ("hmm", "okay so" do not trigger the LLM).

### 5.2 Speech to text

- `faster-whisper` with `small.en`, `beam_size=1`, `initial_prompt` set to the vocabulary list above.
- Post-process: normalize spoken complexity ("oh of n squared", "big O of n") into `O(n^2)` with a few regexes, so complexity detection later is trivial.
- Upgrade path if latency is the problem: Moonshine streaming models (the project reports far lower live-transcription latency than Whisper Small on the same hardware) or Parakeet.

### 5.3 LLM runtime and prompts

**Runtime:** llama.cpp `llama-server` with Qwen3.5-4B (Q4_K_M GGUF) plus its vision projector (`--mmproj`). I chose it over Ollama because Unsloth's Qwen3.5 docs say Qwen3.5 GGUFs with separate mmproj vision files did not work in Ollama at the time of writing, while other write-ups say `ollama run qwen3.5:4b` handles screenshots. Sources disagree, so llama-server is the safe default and Ollama is worth a quick retest.

```bash
llama-server \
  -m <path>/Qwen3.5-4B-Q4_K_M.gguf \
  --mmproj <path>/mmproj-F16.gguf \
  -c 16384 --parallel 2 -ngl 99 --port 8080 --jinja \
  --chat-template-kwargs '{"enable_thinking":false}'
```

(Exact file names depend on which GGUF repo I download from.) `--parallel 2` gives two slots so the interviewer prompt and the observer prompt do not keep evicting each other's cache; llama-server picks the slot with the best prefix match on its own. If I see cache thrashing, pin slots explicitly.

**Thinking mode must be off for conversation.** Qwen3.5 is a hybrid reasoning family and sources disagree on whether the small models think by default (Unsloth says off by default for 0.8B to 9B; another guide says on). Do not rely on the default: set it explicitly and verify that the first streamed token is not reasoning. Also add an output sanitizer that strips `<think>...</think>`, a stray `</think>`, asterisks, markdown, and emoji before text reaches TTS, because there are reports of a dangling `</think>` leaking out even with thinking disabled in some llama.cpp builds. For the end-of-interview judge pass, thinking can be turned back on since latency does not matter there (per request via `chat_template_kwargs`; if the build ignores that, run the judge without it).

**Tag protocol.** Every interviewer reply begins with exactly one control tag, then the spoken text. The Brain buffers the first few tokens, reads the tag, and decides what to do:

| Tag | Meaning | Speaks? | Logged for scoring as |
|---|---|---|---|
| `[PASS]` | Candidate is thinking aloud, say nothing | no | nothing |
| `[ACK]` | Tiny acknowledgement | yes | nothing |
| `[ANSWER]` | Answers a clarifying question about the problem | yes | clarification answered |
| `[PROBE]` | Asks me to explain reasoning | yes | probe |
| `[CHALLENGE]` | Raises a possible flaw as a question | yes | challenge |
| `[HINT1]` `[HINT2]` `[HINT3]` | Nudge, direction, pattern/near-solution | yes | hint at that level |
| `[WRAP]` | Ends the interview | yes | wrap |

The tag doubles as free telemetry for the scorecard (how many hints, at what level, at what time).

**System prompt skeleton:**

```
You are a software engineer running a {minutes} minute coding interview, speaking out loud.
Rules:
- Reply in at most 2 short sentences. No lists, markdown, code, or emojis.
- Begin every reply with exactly one tag from: [PASS] [ACK] [ANSWER] [PROBE] [CHALLENGE] [HINT1] [HINT2] [HINT3] [WRAP].
- Prefer [PASS] while the candidate is coding or talking through their own work.
- Never reveal the solution and never say an approach is optimal. Ask questions instead of correcting.
- If you are unsure whether the candidate is wrong, ask them to walk through an example.
- Neutral, professional tone. No gushing praise.
- You may only give a hint at level {allowed_hint_level} or below.
PROBLEM: {problem_text}
PRIVATE NOTES: {interviewer_notes}
--- state (changes every turn) ---
phase={phase} elapsed={mm:ss} hints_given={n} last_run={result}
LATEST CODE:
{code_snapshot}
```

**Context management:** keep the prompt under about 4k tokens: static prefix, then state, then the last 8 to 10 dialogue turns. The full transcript lives in the Session object, not in the prompt.

### 5.4 TTS

```python
class TTSBackend(Protocol):
    sample_rate: int
    def synth(self, text: str) -> np.ndarray: ...
```

| Backend | Use when | Notes |
|---|---|---|
| `KokoroTTS` (default) | Always works, fast | 82M parameters, Apache-2.0, runs on CPU or GPU, `pip install kokoro` plus system `espeak-ng`. Its API wrapper README reports first-chunk latency of roughly 300 ms on an NVIDIA GPU |
| `OpenAICompatTTS` | I want VoiceStudio (or Kokoro-FastAPI) | `POST /v1/audio/speech` with a Bearer key, returns a WAV |
| `KittenTTS` | CPU-only emergency | Small and English only |

**An honest note on VoiceStudio.** It is a speech workbench (Python backend plus web and Electron UI) wrapping engines like OmniVoice, with voice cloning and design. It is great for sound quality and a custom voice, but for a live conversation it has caveats:

- A third-party install test measured a warm render of about 7.6 seconds of audio in roughly 2.1 to 2.3 seconds on an RTX 3060, and almost 4 minutes on an 8-core CPU-only box. The API returns a finished WAV rather than a stream, so first audio waits for the whole sentence. That is workable on a GPU with short sentences, and unusable on CPU (except its small KittenTTS engine).
- The Docker image is about 17 GB unpacked and the server wants an API key even on localhost.
- The LLM, STT, and TTS would share one GPU.

So the plan is: **build with Kokoro, keep VoiceStudio as a drop-in backend**, and only switch if the voice quality gain is worth the extra latency. Because all replies are 1 to 2 sentences, sentence-level synthesis keeps the penalty bounded.

**Streaming playback:** the Speaker thread pulls sentences from a queue, synthesizes, and writes to a `sounddevice.OutputStream`. A `stop()` flag checked between audio blocks gives instant barge-in.

**Sentence splitter** (flush rule for streamed LLM text):

```python
def should_flush(buf: str) -> bool:
    words = len(buf.split())
    if words >= 5 and buf.rstrip().endswith((".", "?", "!")):
        return True
    if words >= 12 and buf.rstrip().endswith((",", ";", ":")):
        return True      # long sentence: speak the first clause early
    return False
```

### 5.5 Screen understanding

**Capture:** `mss` grabs the configured monitor; Pillow downsizes (max width about 1280 for full screen, up to about 1568 on a tight editor crop, because small models read code badly at low resolution). A `--calibrate` flag saves the crops to disk so I can check them once.

**Change detection (cheap, no model):** shrink each frame to 64x36 grayscale and compare with the previous one. Only treat the screen as "changed" when the mean difference passes a threshold, and only call the VLM once the screen has been stable for about 2 seconds (I stopped typing).

**Two uses of vision:**

1. **At "start" (one shot, full screen, best quality).** Ask the VLM for JSON: `{title, statement, examples, constraints, language}`. Then a text-only call generates **interviewer notes**: brute force and its complexity, optimal approach and complexity, key edge cases, three graded hints, one follow-up. These are private and go into the prompt prefix. Mask the 5 to 15 second delay by saying "Okay, give me a moment to pull up the problem."
2. **Ongoing observer (editor crop, low priority).** Returns JSON: `{code, run_status, failing_case}` where `run_status` is one of `none | running | accepted | wrong_answer | runtime_error | time_limit | compile_error`. Validate the JSON, retry once, otherwise keep the last good snapshot. The observer **skips its turn when the interviewer LLM is busy**, so vision never slows down a reply.

**Reliability reality check.** A 4B vision model transcribing 30 lines of code from a screenshot will make mistakes (especially Python indentation). Mitigations, in order of effort:

1. Crop tight to the editor and keep resolution high.
2. Treat the snapshot as approximate: prompt rules say "do not assert a syntax or typo error unless the same text appears in two consecutive snapshots".
3. Optional later upgrade: OCR the editor crop (for example RapidOCR) and merge with the VLM output.
4. Optional: `problems/<slug>.json` hand-written notes that override the model-generated interviewer notes, for problems I already know well (handy for a NeetCode list). A wrong "optimal solution" from the model is the worst failure mode for a mock interviewer.

### 5.6 The Brain: phases, triggers, and behavior

**Phase machine** (owned by code, not the LLM):

```
IDLE -> INTRO -> CLARIFY -> APPROACH -> CODING -> TESTING -> FOLLOWUP -> WRAPUP -> REPORT
```

| Transition | Detected by |
|---|---|
| IDLE to INTRO | "start" or Enter |
| INTRO to CLARIFY | Interviewer finished stating the problem |
| CLARIFY to APPROACH | I describe an approach (keywords like "brute force", "I would", "approach") or 5 minutes pass |
| APPROACH to CODING | Editor line count grows past a threshold, or I say "let me code this" |
| CODING to TESTING | I say "done", "ready to test", or the observer sees a run |
| TESTING to FOLLOWUP | Accepted result, or I say I'm confident, or the time budget for coding is gone |
| FOLLOWUP to WRAPUP | One follow-up answered, or time up |
| any to WRAPUP | "end interview", hard time limit |

Phases mostly change **thresholds** (silence tolerance, how chatty the interviewer is) and what the interviewer is allowed to probe. Missed transitions are survivable: the clock forces progress.

**Proactive triggers** (the "butting in" behavior). The `Clock` thread emits these as events; the Brain passes them to the LLM as a `TRIGGER:` line so the wording stays natural.

| Trigger | Condition | Typical interviewer line | Cooldown |
|---|---|---|---|
| Dead air, approach | 25 s of no speech, no code | "Where are you at with this?" | 30 s |
| Dead air, coding | 60 s no speech while the editor keeps changing | "Can you talk me through what you're writing?" | 60 s |
| Code before plan | 4 or more new lines and no approach statement logged | "Before you go further, can you walk me through your approach?" | once |
| No complexity | I say I'm done, or a run happens, and no complexity was stated | "What is the time and space complexity?" | once |
| Done without testing | "Done" with no test cases mentioned | "How would you convince yourself this works?" | once |
| Failing run | Observer sees wrong answer, runtime error, or time limit | "What does that failing case tell you?" | per event |
| Long monologue | More than 30 s of continuous speech, then a 1.2 s pause | "Quick question on that last part..." | 45 s |
| Time checks | 50 percent and 80 percent of the session | Brief time status | once each |

Global rule: at most one proactive interjection per 30 seconds, and never while I am speaking. The one place a human interviewer truly interrupts mid-sentence is not feasible with half-duplex audio, so the "long monologue" trigger interrupts at the next micro-pause instead.

**Hint ladder** (enforced by code, not trust): the orchestrator tracks `hints_given` and passes `allowed_hint_level = last_level + 1`. The LLM cannot jump to a near-solution without the earlier levels having been used, unless I ask directly. Levels: 1 nudge as a question, 2 direction (data structure or property to consider), 3 pattern or near-solution outline. Each hint is logged with a timestamp.

**Interviewer behaviors encoded in the prompt and triggers** (from section 2.4): states the problem briefly, answers clarifications tersely, starts with brute force, asks complexity, does not confirm correctness early, points at bugs by asking me to run an example, one follow-up, keeps time, neutral tone.

---

## 6. Session state and scoring (no database)

### 6.1 Session object

One `InterviewSession` dataclass lives in memory for the run:

```python
@dataclass
class InterviewSession:
    started_at: float
    problem: Problem                      # title, statement, notes
    turns: list[Turn]                     # speaker, text, t_start, t_end, phase
    code_snapshots: list[Snapshot]        # t, code, only when changed
    runs: list[RunEvent]                  # t, status, failing_case
    interviewer_events: list[Event]       # t, tag (HINT2, PROBE...), text
    phase_log: list[tuple[str, float]]    # phase name, t entered
```

Process exits, data is gone. `--save-report path.md` writes the final report once if I want to keep one.

### 6.2 Deterministic metrics (computed in plain Python)

| Metric | Source |
|---|---|
| Time to first clarifying question, number of clarifications | turns plus regex for question forms in CLARIFY |
| Time spent per phase | `phase_log` |
| Time to first line of code | `code_snapshots` |
| Talk ratio during CODING and longest silence | turn timestamps |
| Words per minute and filler rate ("um", "uh", "like", "you know") | transcript |
| Hints given, by level, and time of first hint | interviewer tags |
| Complexity stated (time, space), brute force stated before optimal | normalized `O(...)` regex plus keyword spotting |
| Run count, wrong answers, final status | `runs` |
| Test cases and edge cases named | LLM checklist (below), backed by quotes |

### 6.3 Rubric checklist (LLM judge with evidence)

After the interview ends, a judge pass runs over the transcript, the final code, and the metrics. It answers **yes/no questions**, each of which must include a short transcript quote as evidence; if no quote is returned, the answer counts as no. This guards against a small model hallucinating generosity. The judge pass has no latency constraint, so it can use thinking mode, or swap to a larger model (for example Qwen3.5-9B) if the machine can load it after the live session is over.

| Dimension | Checklist items (1 point each) |
|---|---|
| **Communication** | Asked at least one clarifying question before coding; restated the problem or an example; explained the approach before coding; narrated while coding (talk ratio and no silence over 60 s); discussed trade-offs; responded well to hints or challenges |
| **Problem solving** | Stated brute force with complexity; reached optimal or near-optimal (judged against interviewer notes); stated correct time and space complexity; needed no hint above the allowed level for my bar; considered a second approach; handled the follow-up |
| **Technical competency** | Final code ran without compile errors; solved the problem (accepted, or correct by dry run); at most one wrong submission; readable names and structure; idiomatic use of the language and data structures; finished within time |
| **Testing** | Proposed at least two test cases; named at least two real edge cases; walked through code on an example; found own bug before the interviewer pointed it out (or no bug existed); did not declare done without testing |

Scoring a dimension: points to band, `0-1 -> 1`, `2-3 -> 2`, `4-5 -> 3`, `6 -> 4`. Hard overrides from the research: declaring done without any testing caps Testing at 1; never explaining anything while coding caps Communication at 1.

### 6.4 Verdict

```python
def verdict(scores: dict[str, int]) -> str:
    avg = sum(scores.values()) / len(scores)
    lo = min(scores.values())
    if avg < 2.0:                          return "Strong no hire"
    if lo == 1 or avg < 2.5:               return "Leaning no hire"
    if avg < 3.0:                          return "Leaning hire"
    if avg >= 3.5 and lo >= 3:             return "Strong hire"
    return "Hire"
```

Thresholds are constants in `scoring.py`, keyed by bar profile (`intern`, `new_grad`, `senior`). I will tune them by hand after the first few sessions. Caveat: this is a rough self-calibration tool, not an oracle; the deterministic metrics are the trustworthy part, the LLM judgments are a second opinion.

### 6.5 Terminal report

Printed with `rich` at the end of the session:

1. Verdict and the four dimension scores (1 to 4) with one line of evidence each.
2. Timeline of the interview (phase durations, first code, first run, hints with timestamps).
3. Metrics table (talk ratio, longest silence, filler rate, runs, hints by level).
4. Three strengths and three things to fix, each tied to a quote or timestamp.
5. The final code snapshot and the interviewer's private optimal approach, so I can compare.

---

## 7. Project layout, config, CLI

```
mock-interviewer/
  main.py            # CLI entry, wires threads and queues
  config.toml        # all tunables
  audio_in.py        # mic, Silero VAD, utterances, barge-in
  stt.py             # faster-whisper wrapper, text normalization
  llm.py             # llama-server client, streaming, tag parser, sanitizer
  tts.py             # TTSBackend, Kokoro, OpenAICompat, speaker thread, clips
  screen.py          # mss capture, diff, observer thread, VLM prompts
  brain.py           # phase machine, triggers, hint ladder, prompt assembly
  session.py         # dataclasses and metrics
  scoring.py         # checklist, bands, verdict, bar profiles
  report.py          # rich output
  prompts/           # persona.txt, problem_extract.txt, notes.txt, judge.txt
  problems/          # optional hand-written notes per problem (slug.json)
```

Around ten files, a few hundred lines each. That is the whole project.

**`config.toml` sketch:**

```toml
[session]
minutes = 35
bar = "intern"            # intern | new_grad | senior

[audio]
input_device = "default"
end_silence_ms = 700
end_silence_coding_ms = 1500
barge_in_min_ms = 300

[llm]
base_url = "http://127.0.0.1:8080"
max_tokens = 90
temperature = 0.6

[tts]
backend = "kokoro"        # kokoro | openai_compat | kitten
voice = "af_heart"
# openai_compat settings, used for VoiceStudio:
# url = "http://127.0.0.1:3900/v1/audio/speech"
# api_key_env = "OMNIVOICE_API_KEY"
# model = "omnivoice"

[screen]
monitor = 1
editor_region = [0.5, 0.1, 1.0, 0.9]   # fractions: left, top, right, bottom
max_width = 1280
settle_seconds = 2
```

**CLI:**

```
python main.py                     # normal run, say "start" when LeetCode is on screen
python main.py --text-mode         # type instead of speak, print instead of TTS (debug the brain)
python main.py --calibrate         # save screen crops and exit
python main.py --minutes 25 --bar new_grad
python main.py --save-report out.md
```

`--text-mode` is the most useful dev feature: it lets me tune prompts, triggers, and scoring without talking out loud for every test.

---

## 8. Build plan

Each milestone has an exit check. Do them in order.

| # | Milestone | Exit check |
|---|---|---|
| 0 | **Latency smoke tests.** Run each piece alone: llama-server first token and tokens per second (text and one screenshot), faster-whisper on a 5 second clip, Kokoro first audio, optionally VoiceStudio endpoint | A table of real numbers; pick hardware profile A, B, or C and the model sizes |
| 1 | **Voice loop, no screen.** Mic to VAD to STT to LLM to TTS with streaming, sentence splitter, barge-in, and the tag protocol with a generic interviewer prompt | I can chat naturally and interrupt it; end-of-speech to first audio is about 1.5 s or better |
| 2 | **Start and problem reading.** "start" captures the screen, extracts the problem, generates interviewer notes, and the interviewer states the problem aloud | Works on 5 different LeetCode problems; notes look sane |
| 3 | **Interviewer behavior.** Phase machine, triggers, backchannel clips, hint ladder, time checks | A full run in `--text-mode` and by voice feels like an interview, not a chatbot |
| 4 | **Observer.** Background code reading and run-status detection | Code snapshot is usable on most problems; run results detected; no added reply latency |
| 5 | **Session log and scoring.** Metrics, judge pass, verdict, rich report | A run on a problem I solve well and one I deliberately fail give clearly different verdicts |
| 6 | **Polish.** Calibration flag, config cleanup, prompt tuning, optional OCR merge, optional VoiceStudio backend | Three clean end-to-end sessions in a row |

**Install starting point:**

```bash
pip install sounddevice numpy silero-vad faster-whisper kokoro soundfile mss pillow httpx rich
# system packages: portaudio and espeak-ng (brew install portaudio espeak-ng, or apt install)
# llama.cpp: build or download a release that includes llama-server
```

---

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 4B model gives wrong guidance on a problem | Misleading mock feedback | Ask-don't-assert prompt rules; optional `problems/<slug>.json` notes; `[CHALLENGE]` phrased as questions |
| VLM misreads my code | False "you have a typo" interjections | Crop and high resolution; only raise syntax concerns when seen in two consecutive snapshots; optional OCR merge |
| Latency creeps above about 2 s | Feels like a bad phone call, not an interview | Milestone 0 numbers, shorter replies, backchannel clips, drop to 2B, smaller TTS |
| GPU memory contention (LLM plus STT plus TTS) | Crashes or stalls | Put STT and TTS on CPU first; load order at startup; smaller quantization |
| Interviewer voice feeds back into the mic | False barge-ins | Headphones; higher VAD threshold while the interviewer speaks |
| Qwen3.5 thinking tokens leak into speech | Interviewer reads its reasoning aloud | Explicit `enable_thinking=false`, sanitizer in the stream path, a startup self-test |
| Interviewer talks too much | Unrealistic, annoying | `[PASS]` default during coding, cooldowns, global one-interjection-per-30-s rule |
| Judge pass is too generous or harsh | Misleading hire verdict | Evidence-required yes/no checklist, deterministic overrides, optional larger judge model, hand-tuned thresholds |
| STT mishears "start" | Cannot begin | Enter key fallback |
| VoiceStudio too slow live | Awkward pauses | It is an optional backend; Kokoro stays the default |

---

## 10. Stretch ideas (only if the basics feel good)

- A second problem type: a short behavioral round using the same voice loop and a STAR-format checklist.
- Voice cloning in VoiceStudio to make the interviewer sound like a specific person, if the latency is acceptable on a GPU.
- Problem selection by voice ("give me a medium on graphs"), with a local list of problems to read from.
- Merge OCR output with the VLM for more accurate code reading.
- A "tough interviewer" setting that lowers hint generosity and raises probe frequency.

---

## 11. Sources

- Tech Interview Handbook, coding interview rubrics (four dimensions, bands, scoring): https://www.techinterviewhandbook.org/coding-interview-rubrics/
- Prepfully, Google SWE coding interview (Algorithms, Coding, Communication, Problem Solving, 20 minute pacing rule): https://prepfully.com/interview-guides/google-software-engineer-coding-interview
- Prepfully, Meta SWE coding interview (Problem Solving, Coding, Verification, Communication): https://prepfully.com/interview-guides/meta-swe-coding-interview
- Interview behavior and unstuck guidance: https://dev.to/alex_hunter_44f4c9ed6671e/what-to-do-when-completely-stuck-during-a-coding-interview-a-survival-guide-49na and https://levelop.dev/blog/how-to-solve-coding-interview-problems-when-stuck
- Qwen3.5 small models (0.8B, 2B, 4B, 9B, native vision, Apache 2.0): https://insiderllm.com/guides/qwen-3-5-small-models-9b-beats-30b/
- Unsloth Qwen3.5 guide (thinking toggle, sizing, Ollama and mmproj note): https://unsloth.ai/docs/models/qwen3.5.md
- Thinking mode switches across runtimes: https://www.besthub.dev/articles/how-to-turn-thinking-mode-on-or-off-for-qwen3-5-models-in-ollama-lm-studio-llama-cpp-and-vllm-e303d618cd77
- VoiceStudio install and hardware test (OpenAI-compatible endpoint, GPU and CPU timings, licensing): https://computingforgeeks.com/install-voicestudio/
- VoiceStudio by MSR (multi-engine local studio): https://github.com/msrbuilds/voice-studio/releases/tag/v0.2.0
- Kokoro-FastAPI (first-chunk latency by hardware, streaming): https://github.com/StartUpNationLabs/Kokoro-FastAPI
- Moonshine vs Whisper for live speech: https://moonshine-voice.readthedocs.io/en/latest/moonshine-vs-whisper/
- Local speech pipeline reference (Silero VAD, Parakeet, small Qwen, about 700 ms end to end): https://se.linkedin.com/in/andreashogstrom
- Modular VAD, STT, LLM, TTS pipeline for reference: https://pypi.org/project/speech-to-speech/0.2.11/
