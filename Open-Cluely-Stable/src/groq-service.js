const GROQ_API = 'https://api.groq.com/openai/v1/chat/completions';

/**
 * Detailed expert system prompt injected into every Groq API call.
 * Instructs the model to always produce thorough, structured, example-rich answers.
 */
const DETAIL_SYSTEM_PROMPT = `You are an expert AI assistant with 20+ years of combined knowledge across software engineering, data science, mathematics, science, business analysis, and general knowledge.

For EVERY answer you provide, you MUST:
1. Be DETAILED and THOROUGH — never give a one-liner when depth is warranted.
2. Be WELL-STRUCTURED — use ## headings, bullet points, numbered steps, and \`\`\`code blocks\`\`\` where appropriate.
3. EXPLAIN the WHY — don't just state facts; explain reasoning, trade-offs, and context.
4. Include EXAMPLES — give concrete code snippets, real-world analogies, or step-by-step walkthroughs.
5. Cover EDGE CASES — mention common pitfalls, gotchas, and best practices.
6. Use RICH FORMATTING — **bold** key terms, \`inline code\` for technical tokens, and clear section separators.

For CODING questions:
- Provide complete, runnable code with inline comments.
- State time and space complexity.
- Show usage examples.
- Mention alternative approaches.

For CONCEPTUAL questions:
- Explain from first principles.
- Give comparisons and real-world use cases.
- Provide a concise summary at the end.

Do NOT truncate your response. Aim for quality, depth, and completeness in every reply.`;

/**
 * Groq (OpenAI-compatible) backend — same surface as GeminiService for main process swaps.
 */
class GroqService {
  constructor(apiKey, options = {}) {
    this.apiKey = apiKey;
    this.textModel =
      options.textModel ||
      process.env.GROQ_MODEL ||
      'llama-3.3-70b-versatile';
    this.visionModel =
      options.visionModel ||
      process.env.GROQ_VISION_MODEL ||
      'meta-llama/llama-4-scout-17b-16e-instruct';
    this.conversationHistory = [];
    this.model = { provider: 'groq' };
  }

  addToHistory(role, content) {
    this.conversationHistory.push({ role, content });
  }

  clearHistory() {
    this.conversationHistory = [];
  }

  getContextString() {
    return this.conversationHistory.map((e) => `${e.role}: ${e.content}`).join('\n\n');
  }

  async _chat(model, messages) {
    // Always prepend the expert detail system prompt
    const withSystem = [
      { role: 'system', content: DETAIL_SYSTEM_PROMPT },
      ...messages
    ];

    const res = await fetch(GROQ_API, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages: withSystem,
        temperature: 0.65,
        max_tokens: 4096,
      }),
    });

    const raw = await res.text();
    if (!res.ok) {
      throw new Error(`Groq API ${res.status}: ${raw}`);
    }
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`Groq API: invalid JSON response`);
    }
    const text = data.choices?.[0]?.message?.content;
    if (text == null || text === '') {
      throw new Error('Groq API: empty response');
    }
    return text;
  }

  async generateText(prompt) {
    return this._chat(this.textModel, [{ role: 'user', content: prompt }]);
  }

  /**
   * @param {Array<string|{inlineData:{data:string,mimeType:string}}>} parts
   * First entry is text; rest are Gemini-style inline image parts.
   */
  async generateMultimodal(parts) {
    if (!parts?.length) {
      throw new Error('Groq multimodal: no parts');
    }
    const head = parts[0];
    const promptText = typeof head === 'string' ? head : head?.text || '';
    const content = [{ type: 'text', text: promptText }];

    for (let i = 1; i < parts.length; i++) {
      const p = parts[i];
      const inline = p?.inlineData;
      if (!inline?.data || !inline?.mimeType) continue;
      const url = `data:${inline.mimeType};base64,${inline.data}`;
      content.push({
        type: 'image_url',
        image_url: { url },
      });
    }

    if (content.length === 1) {
      return this._chat(this.textModel, [{ role: 'user', content: promptText }]);
    }

    return this._chat(this.visionModel, [{ role: 'user', content }]);
  }

  async suggestResponse(context) {
    const prompt = `
You are an AI assistant helping in a meeting.

Context:
${context}

Provide 3 concise suggestions.
`;
    return await this.generateText(prompt);
  }

  async generateMeetingNotes() {
    const prompt = `
Create professional meeting notes from this conversation:

${this.getContextString()}
`;
    return await this.generateText(prompt);
  }

  async generateFollowUpEmail() {
    const prompt = `
Write a professional follow-up email based on this conversation:

${this.getContextString()}
`;
    return await this.generateText(prompt);
  }

  async answerQuestion(question) {
    const prompt = `
Conversation context:

${this.getContextString()}

Question:
${question}
`;
    const result = await this.generateText(prompt);
    this.addToHistory('user', question);
    this.addToHistory('assistant', result);
    return result;
  }

  async getConversationInsights() {
    const prompt = `
Analyze this conversation and provide insights:

${this.getContextString()}
`;
    return await this.generateText(prompt);
  }
}

module.exports = GroqService;
