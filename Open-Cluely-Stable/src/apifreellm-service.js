/**
 * ApiFreeLLM Service
 * Handles integration with apifreellm.com free tier API.
 */
class ApiFreeLLMService {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.apiUrl = 'https://apifreellm.com/api/v1/chat';
    this.conversationHistory = [];
    this.model = { provider: 'apifreellm' };
  }

  addToHistory(role, content) {
    this.conversationHistory.push({ role, content });
  }

  clearHistory() {
    this.conversationHistory = [];
  }

  getContextString() {
    return this.conversationHistory
      .map((e) => `${e.role}: ${e.content}`)
      .join('\n\n');
  }

  async _request(message) {
    const res = await fetch(this.apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        message: message,
        model: 'apifreellm',
      }),
    });

    const raw = await res.text();
    if (!res.ok) {
      throw new Error(`ApiFreeLLM API ${res.status}: ${raw}`);
    }

    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error(`ApiFreeLLM API: invalid JSON response`);
    }

    if (data.success && data.response) {
      return data.response;
    } else {
      throw new Error(data.error || 'ApiFreeLLM API: request failed');
    }
  }

  async generateText(prompt) {
    return this._request(prompt);
  }

  /**
   * ApiFreeLLM free tier does not support vision/multimodal.
   */
  async generateMultimodal(parts) {
    throw new Error(
      'ApiFreeLLM free tier does not support screenshot analysis (Vision). Please use a Gemini or Groq API key for this feature.'
    );
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

module.exports = ApiFreeLLMService;
