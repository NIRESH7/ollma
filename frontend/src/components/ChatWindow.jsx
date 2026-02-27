import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Send, User, Bot, Sparkles, Loader2, Info, RefreshCw, Database } from 'lucide-react';

const ChatWindow = ({ activeFolder }) => {
    const [messages, setMessages] = useState([
        { role: 'bot', content: "Neural RAG ready. Knowledge context loaded from '" + activeFolder + "'. How can I assist you?" }
    ]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isLoading]);

    const handleSend = async (e) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMessage = { role: 'user', content: input };
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);

        try {
            const response = await axios.post('http://127.0.0.1:8000/query/', {
                question: input,
                folder: activeFolder || "All"
            });

            setMessages(prev => [...prev, { role: 'bot', content: response.data.answer }]);
        } catch (error) {
            console.error("Query failed", error);
            setMessages(prev => [...prev, {
                role: 'bot',
                content: "Error: Retrieval node unreachable. Please check backend infrastructure."
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex-1 flex flex-col h-full overflow-hidden relative">
            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto px-10 py-8 space-y-10 custom-scrollbar relative pb-32">
                {messages.map((msg, idx) => (
                    <div
                        key={idx}
                        className={`flex items-start gap-4 animate-fade-in ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                    >
                        {/* Avatar Container */}
                        <div className={`shrink-0 flex flex-col items-center gap-2 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                            <div className={`p-2.5 rounded-xl border transition-all duration-500 ${msg.role === 'user'
                                ? 'bg-indigo-600 text-white border-white/20 shadow-xl shadow-indigo-500/40'
                                : 'bg-white/60 text-indigo-500 border-white/80 shadow-sm'
                                }`}>
                                {msg.role === 'user' ? <User className="w-5 h-5" /> : <Database className="w-5 h-5" />}
                            </div>
                        </div>

                        {/* Message Content */}
                        <div className={`max-w-[80%] space-y-2.5 ${msg.role === 'user' ? 'text-right' : ''}`}>
                            <div className={`px-7 py-4 rounded-[24px] text-[15px] leading-relaxed transition-all ${msg.role === 'user'
                                ? 'chat-bubble-glow text-white font-medium'
                                : 'bot-bubble text-slate-700 font-medium'
                                }`}>
                                {msg.content}
                            </div>
                            <p className="text-[10px] font-black text-slate-600 uppercase tracking-[0.4em] px-3">
                                {msg.role === 'user' ? 'Neural Operator' : 'Knowledge Engine'}
                            </p>
                        </div>
                    </div>
                ))}

                {isLoading && (
                    <div className="flex items-start gap-4 animate-pulse">
                        <div className="p-2.5 rounded-xl bg-white/60 text-indigo-500 border border-white/80 shadow-sm">
                            <RefreshCw className="w-5 h-5 animate-spin" />
                        </div>
                        <div className="px-7 py-4 rounded-[24px] bot-bubble flex items-center gap-6">
                            <div className="flex gap-2">
                                <div className="w-2 h-2 rounded-full bg-indigo-500/60 animate-bounce"></div>
                                <div className="w-2 h-2 rounded-full bg-indigo-500/60 animate-bounce [animation-delay:0.2s]"></div>
                                <div className="w-2 h-2 rounded-full bg-indigo-500/60 animate-bounce [animation-delay:0.4s]"></div>
                            </div>
                            <span className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.3em]">PROCESSING CONTEXT...</span>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            {/* Input Area - Floating Glass */}
            <div className="absolute bottom-8 left-0 right-0 px-10 z-20">
                <form
                    onSubmit={handleSend}
                    className="max-w-3xl mx-auto input-panel p-2 rounded-[20px] flex items-center gap-3 focus-within:ring-1 focus-within:ring-white/20 transition-all"
                >
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Type your message..."
                        className="flex-1 bg-transparent border-none focus:ring-0 text-[14px] font-medium text-slate-800 placeholder:text-slate-400 px-5 py-3"
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !input.trim()}
                        className="p-3 bg-indigo-600 text-white rounded-[14px] hover:bg-indigo-500 disabled:opacity-20 transition-all flex items-center justify-center shadow-lg shadow-indigo-500/40 active:scale-95 group"
                    >
                        {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />}
                    </button>
                </form>
            </div>
        </div>



    );
};

export default ChatWindow;
