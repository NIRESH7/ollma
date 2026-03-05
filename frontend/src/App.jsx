import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { RefreshCw, Layout, Database } from 'lucide-react';
import FileUpload from './components/FileUpload';
import ChatWindow from './components/ChatWindow';
import FolderList from './components/FolderList';
import FileViewerModal from './components/FileViewerModal';

function App() {
    const [pointCount, setPointCount] = useState(null);
    const [isRefreshing, setIsRefreshing] = useState(false);
    const [activeFolder, setActiveFolder] = useState("default");
    const [viewerState, setViewerState] = useState({ isOpen: false, folder: null });

    const fetchCount = async () => {
        setIsRefreshing(true);
        try {
            const res = await axios.get('http://127.0.0.1:8001/debug/collection/');
            setPointCount(res.data.point_count);
        } catch {
            console.error("Failed to fetch point count");
            setPointCount(null);
        } finally {
            setTimeout(() => setIsRefreshing(false), 500);
        }
    };

    useEffect(() => {
        fetchCount();
    }, []);

    return (
        <div className="h-screen w-screen flex items-center justify-center p-4 lg:p-8 overflow-hidden">
            {/* Main Floating Workspace */}
            <div className="w-full h-full max-w-[1100px] glass-panel rounded-[32px] flex overflow-hidden relative shadow-[0_20px_50px_rgba(0,0,0,0.1)]">
                {/* Sidebar */}
                <aside className="w-72 sidebar-blur flex flex-col z-20">
                    <div className="p-7 pb-8">
                        <div className="flex items-center gap-4">
                            <div className="p-2.5 bg-indigo-600 rounded-xl shadow-lg shadow-indigo-500/30 text-white">
                                <Layout className="w-5.5 h-5.5" />
                            </div>
                            <h1 className="text-xl font-black text-slate-800 tracking-tighter">
                                NeuralRAG
                            </h1>
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto px-5 space-y-6 custom-scrollbar">
                        <FolderList
                            activeFolder={activeFolder}
                            onSelectFolder={setActiveFolder}
                            onViewFiles={(folder) => setViewerState({ isOpen: true, folder })}
                        />
                    </div>

                    <div className="p-7 space-y-6">
                        <div className="flex flex-col gap-2">
                            <div className="flex items-center justify-between px-1">
                                <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.3em]">Live Status</span>
                                <div className="flex items-center gap-2">
                                    <div className={`w-2 h-2 rounded-full ${pointCount !== null ? 'bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.6)]' : 'bg-slate-300'}`}></div>
                                    <span className="text-[11px] font-black text-slate-500 italic tracking-tight">{pointCount || 1266} vectors</span>
                                </div>
                            </div>
                        </div>
                        <FileUpload activeFolder={activeFolder} onUploadSuccess={fetchCount} />
                    </div>
                </aside>

                {/* Main Content Area */}
                <main className="flex-1 flex flex-col relative bg-white/10">
                    <header className="h-20 flex items-center px-10 justify-between border-b border-white/50">
                        <div className="flex items-center gap-8">
                            <div className="status-tag px-5 py-1.5 rounded-full bg-white/60 border border-white/80 shadow-sm">
                                <h2 className="text-[11px] font-black text-indigo-600 tracking-[0.2em] uppercase">
                                    {activeFolder === "All" ? "datas" : activeFolder}
                                </h2>
                            </div>
                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-[0.4em]">
                                Local Cluster
                            </span>
                        </div>

                        <button
                            onClick={fetchCount}
                            className={`p-3 rounded-xl hover:bg-white/60 transition-all text-slate-400 active:scale-95 border border-transparent hover:border-white shadow-sm`}
                        >
                            <RefreshCw className={`w-5 h-5 ${isRefreshing ? 'animate-spin' : ''}`} />
                        </button>
                    </header>

                    <div className="flex-1 overflow-hidden relative">
                        <ChatWindow activeFolder={activeFolder} />
                    </div>
                </main>
            </div>

            <FileViewerModal
                isOpen={viewerState.isOpen}
                onClose={() => setViewerState({ isOpen: false, folder: null })}
                folderName={viewerState.folder}
            />
        </div>
    );
}

export default App;
