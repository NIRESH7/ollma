import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Folder, Plus, ChevronRight, Hash, Eye, Trash2, Layout } from 'lucide-react';

const FolderList = ({ activeFolder, onSelectFolder, onViewFiles, onDeleteFolder }) => {
    const [folders, setFolders] = useState(["default"]);
    const [newFolderName, setNewFolderName] = useState("");
    const [isCreating, setIsCreating] = useState(false);

    const fetchFolders = async () => {
        try {
            const res = await axios.get('http://127.0.0.1:8001/folders');
            setFolders(res.data.folders);
        } catch (error) {
            console.error("Failed to fetch folders", error);
        }
    };

    const handleCreateFolder = async (e) => {
        e.preventDefault();
        if (!newFolderName.trim()) return;

        try {
            await axios.post('http://127.0.0.1:8001/folders', { folder_name: newFolderName });
            setNewFolderName("");
            setIsCreating(false);
            fetchFolders();
        } catch (error) {
            console.error("Failed to create folder", error);
        }
    };

    const handleDeleteFolder = async (e, folderName) => {
        e.stopPropagation();
        if (!window.confirm(`Are you sure you want to delete "${folderName}" and all its contents?`)) return;

        try {
            await axios.delete(`http://127.0.0.1:8001/folders/${folderName}`);
            if (activeFolder === folderName) onSelectFolder("All");
            fetchFolders();
            if (onDeleteFolder) onDeleteFolder(folderName);
        } catch (error) {
            console.error("Failed to delete folder", error);
        }
    };

    useEffect(() => {
        fetchFolders();
    }, []);

    return (
        <div className="space-y-8">
            <div className="flex items-center justify-between px-2">
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.4em] opacity-80">Local Cluster</span>
                <button
                    onClick={() => setIsCreating(!isCreating)}
                    className="p-1.5 rounded-xl hover:bg-white/60 text-slate-400 transition-all hover:text-indigo-600 border border-transparent hover:border-white/80"
                >
                    <Plus className={`w-4 h-4 transition-transform duration-700 ${isCreating ? 'rotate-90' : ''}`} />
                </button>
            </div>

            {isCreating && (
                <form onSubmit={handleCreateFolder} className="animate-fade-in px-2 mb-8">
                    <div className="relative">
                        <input
                            autoFocus
                            type="text"
                            value={newFolderName}
                            onChange={(e) => setNewFolderName(e.target.value)}
                            placeholder="Unit ID..."
                            className="w-full bg-white/40 border border-white/60 rounded-xl px-5 py-3 text-[13px] text-slate-700 focus:outline-none focus:border-indigo-400/50 transition-all placeholder:text-slate-400 font-medium shadow-sm shadow-indigo-100/5"
                        />
                        <button type="submit" className="absolute right-2 top-2 p-1 bg-indigo-600 text-white rounded-lg shadow-lg shadow-indigo-500/20">
                            <ChevronRight className="w-4 h-4" />
                        </button>
                    </div>
                </form>
            )}

            <div className="space-y-0.5">
                {/* Master Index Item */}
                <div className="relative group">
                    <button
                        onClick={() => onSelectFolder("All")}
                        className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${activeFolder === "All"
                            ? 'bg-white/40 text-slate-900 shadow-sm border border-white/60 font-bold'
                            : 'text-slate-500 hover:bg-white/20 hover:text-slate-800'
                            }`}
                    >
                        <div className={`absolute left-0 w-0.5 h-4 rounded-r-full transition-all duration-500 ${activeFolder === "All" ? 'bg-indigo-600 scale-y-100 shadow-[0_0_8px_rgba(99,102,241,0.6)]' : 'bg-transparent scale-y-0'}`}></div>
                        <Layout className={`w-4 h-4 transition-colors ${activeFolder === "All" ? 'text-indigo-600' : 'text-slate-400'}`} />
                        <span className="text-[14px] tracking-tight">Master Index</span>
                    </button>
                </div>

                {folders.map((folder) => (
                    <div key={folder} className="group relative">
                        <button
                            onClick={() => onSelectFolder(folder)}
                            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all ${activeFolder === folder
                                ? 'bg-white/40 text-slate-900 shadow-sm border border-white/60 font-bold'
                                : 'text-slate-500 hover:bg-white/20 hover:text-slate-800'
                                }`}
                        >
                            <div className={`absolute left-0 w-0.5 h-4 rounded-r-full transition-all duration-500 ${activeFolder === folder ? 'bg-indigo-600 scale-y-100 shadow-[0_0_8px_rgba(99,102,241,0.6)]' : 'bg-transparent scale-y-0'}`}></div>
                            <Folder className={`w-4 h-4 transition-colors ${activeFolder === folder ? 'text-indigo-600' : 'text-slate-400'}`} />
                            <span className="text-[14px] tracking-tight truncate">{folder}</span>
                        </button>

                        <div className={`absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 transition-opacity duration-300 ${activeFolder === folder ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                            }`}>
                            <button
                                onClick={(e) => { e.stopPropagation(); if (onViewFiles) onViewFiles(folder); }}
                                className="p-1.5 rounded-lg text-slate-400 hover:text-indigo-600 hover:bg-white/60 transition-all border border-transparent hover:border-white/80"
                            >
                                <Eye className="w-3.5 h-3.5" />
                            </button>
                            <button
                                onClick={(e) => handleDeleteFolder(e, folder)}
                                className="p-1.5 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50/20 transition-all border border-transparent hover:border-white/80"
                            >
                                <Trash2 className="w-3.5 h-3.5" />
                            </button>
                        </div>
                    </div>
                ))}
            </div>
        </div>



    );
};

export default FolderList;
