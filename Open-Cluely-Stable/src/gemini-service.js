const { GoogleGenerativeAI } = require("@google/generative-ai");

class GeminiService {

constructor(apiKey) {

this.genAI = new GoogleGenerativeAI(apiKey);

this.model = this.genAI.getGenerativeModel({
model: "gemini-2.5-flash-lite"
});

this.conversationHistory = [];

}

addToHistory(role, content){
this.conversationHistory.push({role, content});
}

clearHistory(){
this.conversationHistory = [];
}

getContextString(){
return this.conversationHistory
.map(e => `${e.role}: ${e.content}`)
.join("\n\n");
}

async generateText(prompt){

const result = await this.model.generateContent(prompt);

return result.response.text();

}

async generateMultimodal(parts){

const result = await this.model.generateContent(parts);

return result.response.text();

}

async suggestResponse(context){

const prompt = `
You are an AI assistant helping in a meeting.

Context:
${context}

Provide 3 concise suggestions.
`;

return await this.generateText(prompt);

}

async generateMeetingNotes(){

const prompt = `
Create professional meeting notes from this conversation:

${this.getContextString()}
`;

return await this.generateText(prompt);

}

async generateFollowUpEmail(){

const prompt = `
Write a professional follow-up email based on this conversation:

${this.getContextString()}
`;

return await this.generateText(prompt);

}

async answerQuestion(question){

const prompt = `
Conversation context:

${this.getContextString()}

Question:
${question}
`;

const result = await this.generateText(prompt);

this.addToHistory("user", question);
this.addToHistory("assistant", result);

return result;

}

async getConversationInsights(){

const prompt = `
Analyze this conversation and provide insights:

${this.getContextString()}
`;

return await this.generateText(prompt);

}

}

module.exports = GeminiService;