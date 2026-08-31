'use client';

import { useState } from 'react';

const lessons = {
  collectives: {
    title: 'All-reduce foundations',
    goal: 'Combine local rank values into one shared result.',
    result: 'Both ranks print 3.0.',
    steps: [
      ['Local tensors', '[1]', '[2]', 'Each rank owns a different tensor.'],
      ['All-reduce SUM', 'send + receive', 'send + receive', 'Combine all rank values and return the sum everywhere.'],
      ['Shared result', '[3]', '[3]', 'Both ranks now have the same correct answer.'],
    ],
  },
  'tensor-parallel-mlp': {
    title: 'Tensor-parallel MLP',
    goal: 'Split hidden features, compute locally, then sum partial final outputs.',
    result: 'All-reduced output matches the unsharded baseline.',
    steps: [
      ['Replicated input', 'X  (1 × 4)', 'X  (1 × 4)', 'Both ranks need the input for the first linear layer.'],
      ['Column parallel + GELU', 'hidden 0–2  (1 × 3)', 'hidden 3–5  (1 × 3)', 'Each rank owns different hidden features.'],
      ['Row parallel', 'partial Y  (1 × 4)', 'partial Y  (1 × 4)', 'Each rank contributes to every final output feature.'],
      ['All-reduce SUM', 'full Y  (1 × 4)', 'full Y  (1 × 4)', 'Sum partial outputs so every rank receives the final activation.'],
    ],
  },
  'sequence-parallel': {
    title: 'Sequence-parallel layouts',
    goal: 'Move between token ownership and tensor-parallel computation.',
    result: 'Final token shards match the unsharded baseline.',
    steps: [
      ['SP region', 'tokens 1–2 × full hidden', 'tokens 3–4 × full hidden', 'Each rank owns different token positions.'],
      ['G: all-gather', 'all 4 tokens', 'all 4 tokens', 'Both ranks temporarily see the whole sequence for TP work.'],
      ['TP computation', 'partial output', 'partial output', 'Each rank computes its hidden-feature contribution.'],
      ['G*: reduce-scatter', 'tokens 1–2 × full hidden', 'tokens 3–4 × full hidden', 'Sum contributions and return tokens to their owners.'],
    ],
  },
  'pipeline-afab': {
    title: 'Pipeline AFAB',
    goal: 'Move micro-batches through model stages, then backpropagate in reverse.',
    result: 'Both stage gradients match the unpipelined baseline.',
    steps: [
      ['F0', 'stage 0 → send μ0', 'receive μ0 → stage 1', 'The first micro-batch enters the pipeline.'],
      ['F1', 'stage 0 → send μ1', 'receive μ1 → stage 1', 'Keep forwarding micro-batches before backward starts.'],
      ['F2', 'stage 0 → send μ2', 'receive μ2 → stage 1', 'All forwards are complete.'],
      ['B2', 'receive ∂μ2 → backward', 'backward μ2 → send ∂μ2', 'Backward begins at the final stage.'],
      ['B1', 'receive ∂μ1 → backward', 'backward μ1 → send ∂μ1', 'The next gradient moves upstream.'],
      ['B0', 'receive ∂μ0 → backward', 'backward μ0 → send ∂μ0', 'Both stages accumulated gradients for the batch.'],
    ],
  },
  '1f1b': {
    title: '1F1B schedule', goal: 'Alternate forward and backward work after warmup.', result: 'Each stage reports a valid warmup, steady state, and drain.',
    steps: [['Warmup', 'F0 → F1', 'F0', 'Pipeline stages fill from the front.'], ['Steady state', 'F2 → B0', 'F1 → B0', 'Forward and backward work alternate.'], ['Drain', 'B2 → B1', 'B2 → B1', 'Remaining gradients drain backward.']],
  },
  interleaved: {
    title: 'Interleaved pipeline', goal: 'Give each physical stage separated virtual model chunks.', result: 'Each physical stage revisits virtual chunks 0 and 1.',
    steps: [['First virtual pass', 'chunk 0', 'chunk 0', 'The micro-batch crosses the first set of virtual chunks.'], ['Loop back', 'chunk 1', 'chunk 1', 'The same physical ranks process later model chunks.'], ['Reverse pass', 'B chunk 1 → 0', 'B chunk 1 → 0', 'Backward follows the reverse virtual route.']],
  },
} as const;

type LessonId = keyof typeof lessons;

export default function Home() {
  const [id, setId] = useState<LessonId>('tensor-parallel-mlp');
  const [step, setStep] = useState(0);
  const lesson = lessons[id];
  const current = lesson.steps[step];
  const select = (next: LessonId) => { setId(next); setStep(0); };

  return <main className="min-h-screen bg-slate-950 text-slate-100">
    <header className="border-b border-slate-800 px-6 py-5"><div className="mx-auto flex max-w-6xl items-center justify-between gap-4">
      <div><p className="text-xs font-medium tracking-[.24em] text-cyan-300">MINI PICOTRON CPU</p><h1 className="mt-1 text-2xl font-semibold">Distributed training, made inspectable.</h1></div>
      <span className="rounded-full border border-cyan-400/30 bg-cyan-400/10 px-3 py-1 text-sm text-cyan-200">CPU correctness lab</span>
    </div></header>
    <div className="mx-auto grid max-w-6xl gap-8 px-6 py-8 lg:grid-cols-[250px_1fr]">
      <aside><p className="mb-3 text-sm text-slate-400">Learning path</p><nav className="space-y-2">
        {(Object.entries(lessons) as [LessonId, typeof lesson][]).map(([key, value], index) => <button key={key} onClick={() => select(key)} className={`w-full rounded-xl border p-4 text-left ${key === id ? 'border-cyan-300/60 bg-cyan-400/10' : 'border-slate-800 bg-slate-900/40 hover:border-slate-600'}`}><span className="text-xs text-cyan-300">0{index + 1}</span><span className="mt-1 block font-medium">{value.title}</span><span className="mt-1 block text-sm text-slate-400">{value.goal}</span></button>)}
      </nav></aside>
      <section><p className="text-sm text-cyan-300">Interactive lesson</p><h2 className="mt-1 text-3xl font-semibold">{lesson.title}</h2><p className="mt-3 max-w-2xl text-slate-300">{lesson.goal}</p>
        <div className="mt-7 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 sm:p-7"><div className="flex items-center justify-between border-b border-slate-800 pb-5"><div><p className="text-sm text-slate-400">Current operation</p><p className="text-xl font-medium text-cyan-200">{current[0]}</p></div><p className="text-sm text-slate-400">Step {step + 1} / {lesson.steps.length}</p></div>
          <div className="mt-6 grid gap-4 md:grid-cols-2">{[['Rank 0', current[1]], ['Rank 1', current[2]]].map(([rank, action]) => <div key={rank} className="rounded-xl border border-slate-700 bg-slate-950 p-5"><p className="font-medium">{rank}</p><p className="mt-6 font-mono text-sm text-cyan-200">{action}</p></div>)}</div>
          <p className="mt-5 rounded-lg bg-slate-950 px-4 py-3 text-sm text-slate-300">{current[3]}</p>
          <div className="mt-6 flex items-center justify-between"><button onClick={() => setStep(Math.max(0, step - 1))} disabled={!step} className="rounded-lg border border-slate-700 px-4 py-2 text-sm disabled:opacity-40">Previous</button><button onClick={() => setStep(Math.min(lesson.steps.length - 1, step + 1))} disabled={step === lesson.steps.length - 1} className="rounded-lg bg-cyan-300 px-4 py-2 text-sm font-medium text-slate-950 disabled:opacity-40">Next</button></div>
        </div>
        <div className="mt-6 grid gap-4 sm:grid-cols-2"><div className="rounded-xl border border-slate-800 p-5"><p className="text-sm text-slate-400">Run the real experiment</p><code className="mt-3 block rounded-lg bg-slate-900 px-3 py-2 text-sm text-cyan-200">python src/lab.py {id}</code></div><div className="rounded-xl border border-slate-800 p-5"><p className="text-sm text-slate-400">Correctness signal</p><p className="mt-3 text-sm text-slate-200">{lesson.result}</p></div></div>
      </section>
    </div>
  </main>;
}
