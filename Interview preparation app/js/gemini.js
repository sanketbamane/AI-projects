// ai.js - Groq API integration (OpenAI-compatible, free tier)

const GROQ_API_BASE = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_MODEL = 'llama-3.3-70b-versatile'; // Best free Groq model

const Gemini = { // keeping name for compatibility with all pages
  async call(prompt, systemInstruction = '') {
    const apiKey = Storage.getApiKey();
    if (!apiKey) throw new Error('NO_API_KEY');

    const messages = [];
    if (systemInstruction) {
      messages.push({ role: 'system', content: systemInstruction });
    }
    messages.push({ role: 'user', content: prompt });

    const res = await fetch(GROQ_API_BASE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages,
        temperature: 0.8,
        max_tokens: 1024,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      const msg = err?.error?.message || `API error ${res.status}`;
      throw new Error(msg);
    }

    const data = await res.json();
    return data.choices?.[0]?.message?.content || '';
  },

  // ── MOCK INTERVIEW ────────────────────────────────────────────────────────

  async generateInterviewQuestion(role, difficulty, questionNumber, previousQuestions = []) {
    const prevList = previousQuestions.length
      ? `\nDo NOT repeat these questions:\n${previousQuestions.map((q, i) => `${i + 1}. ${q}`).join('\n')}`
      : '';

    const prompt = `You are an expert interviewer at a top tech company.
Generate a single ${difficulty}-level interview question for a ${role} candidate. This is question #${questionNumber} of the session.
The question should be realistic, specific, and thought-provoking.
Mix between technical, behavioral, and situational questions naturally.${prevList}

IMPORTANT: Return ONLY the question itself — no preamble, no numbering, no quotation marks. Just the question.`;

    return await this.call(prompt);
  },

  async evaluateAnswer(question, answer, role, difficulty) {
    const prompt = `You are an expert interviewer evaluating a candidate's answer for a ${role} position (${difficulty} level).

Question: "${question}"
Candidate's Answer: "${answer}"

Evaluate this answer and respond in the following JSON format ONLY (no extra text):
{
  "score": <number from 1-10>,
  "verdict": "<Excellent|Good|Average|Needs Work>",
  "strengths": ["<strength1>", "<strength2>"],
  "improvements": ["<improvement1>", "<improvement2>"],
  "betterAnswer": "<A brief ideal answer in 2-3 sentences>"
}`;

    const raw = await this.call(prompt);
    try {
      const match = raw.match(/\{[\s\S]*\}/);
      return JSON.parse(match ? match[0] : raw);
    } catch {
      return {
        score: 6,
        verdict: 'Good',
        strengths: ['You attempted the question'],
        improvements: ['Try to be more specific with examples'],
        betterAnswer: 'A comprehensive answer would include specific examples from your experience.',
      };
    }
  },

  async generateInterviewSummary(role, qa_pairs) {
    const sessionText = qa_pairs.map((p, i) =>
      `Q${i + 1}: ${p.question}\nA: ${p.answer}\nScore: ${p.score}/10`
    ).join('\n\n');

    const prompt = `You interviewed a candidate for ${role}. Here's the session:

${sessionText}

Provide a brief, constructive overall summary in this JSON format ONLY:
{
  "overallScore": <average score rounded to 1 decimal>,
  "summary": "<2-3 sentence overall assessment>",
  "topStrength": "<their biggest strength>",
  "keySuggestion": "<the most important thing to work on>"
}`;

    const raw = await this.call(prompt);
    try {
      const match = raw.match(/\{[\s\S]*\}/);
      return JSON.parse(match ? match[0] : raw);
    } catch {
      const avg = qa_pairs.reduce((s, p) => s + (p.score || 5), 0) / qa_pairs.length;
      return {
        overallScore: Math.round(avg * 10) / 10,
        summary: 'You completed the mock interview session.',
        topStrength: 'Willingness to practice',
        keySuggestion: 'Keep practicing and reviewing feedback.',
      };
    }
  },

  // ── FLASHCARDS ────────────────────────────────────────────────────────────

  async generateFlashcards(topic, count = 5, difficulty = 'Mid') {
    const prompt = `Generate ${count} high-quality flashcard Q&A pairs for the topic: "${topic}" at ${difficulty} level.

These should be questions commonly asked in technical interviews. Cover different aspects of the topic.

Return ONLY valid JSON in this exact format:
[
  {
    "q": "Question text here?",
    "a": "Clear, concise answer here. Include key points a candidate should mention."
  }
]

No preamble, no extra text — only the JSON array.`;

    const raw = await this.call(prompt);
    try {
      const match = raw.match(/\[[\s\S]*\]/);
      return JSON.parse(match ? match[0] : raw);
    } catch {
      return [{ q: 'Could not generate questions. Check your API key.', a: 'Please verify your Groq API key in Settings.' }];
    }
  },

  // ── RESUME ANALYZER ───────────────────────────────────────────────────────

  async analyzeResume(resumeText, targetRole) {
    const prompt = `You are an expert tech recruiter and ATS specialist. Analyze the following resume for a ${targetRole} position.

Resume:
---
${resumeText}
---

Respond ONLY in this exact JSON format:
{
  "atsScore": <number 0-100>,
  "overallRating": "<Excellent|Strong|Average|Weak>",
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "weaknesses": ["<weakness1>", "<weakness2>"],
  "missingKeywords": ["<keyword1>", "<keyword2>", "<keyword3>", "<keyword4>", "<keyword5>"],
  "suggestions": [
    { "section": "<Section Name>", "issue": "<issue>", "fix": "<specific rewrite suggestion>" }
  ],
  "formatTips": ["<tip1>", "<tip2>"]
}`;

    const raw = await this.call(prompt);
    try {
      const match = raw.match(/\{[\s\S]*\}/);
      return JSON.parse(match ? match[0] : raw);
    } catch {
      return {
        atsScore: 50,
        overallRating: 'Average',
        summary: 'Could not fully parse the response.',
        strengths: ['Resume was submitted'],
        weaknesses: ['Could not analyze properly'],
        missingKeywords: [],
        suggestions: [],
        formatTips: ['Try pasting cleaner text'],
      };
    }
  },

  // ── CODING ────────────────────────────────────────────────────────────────

  async getCodeHint(problem, userCode, hintNumber) {
    const prompt = `A student is solving this coding problem:
"${problem.title}": ${problem.description}

Their current code:
\`\`\`
${userCode || '(no code written yet)'}
\`\`\`

Give hint #${hintNumber} — be progressively more specific with each hint number.
Hint 1: General approach/strategy only.
Hint 2: Which data structure or algorithm to use.
Hint 3: A more specific implementation detail.

Return ONLY the hint text, no extra formatting.`;

    return await this.call(prompt);
  },

  async reviewCode(problem, userCode) {
    const prompt = `Review this code for the problem "${problem.title}":

Problem: ${problem.description}

Code:
\`\`\`javascript
${userCode}
\`\`\`

Provide feedback in this JSON format ONLY:
{
  "isCorrect": <true|false>,
  "timeComplexity": "<e.g. O(n)>",
  "spaceComplexity": "<e.g. O(1)>",
  "feedback": "<2-3 sentence assessment>",
  "improvements": ["<improvement1>", "<improvement2>"]
}`;

    const raw = await this.call(prompt);
    try {
      const match = raw.match(/\{[\s\S]*\}/);
      return JSON.parse(match ? match[0] : raw);
    } catch {
      return {
        isCorrect: false,
        timeComplexity: 'Unknown',
        spaceComplexity: 'Unknown',
        feedback: 'Could not analyze your code. Make sure you have a valid API key.',
        improvements: [],
      };
    }
  },
};
