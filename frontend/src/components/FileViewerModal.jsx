import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { X, FileText, Loader2, Hash, AlertCircle } from 'lucide-react';

const FileViewerModal = ({ isOpen, onClose, folderName }) => {
    const [files, setFiles] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (isOpen && folderName) {
            fetchFiles();
        }
    }, [isOpen, folderName]);

    const fetchFiles = async () => {
        setIsLoading(true);
        setError(null);
        try {
            const res = await axios.get(`http://127.0.0.1:8000/folders/${folderName}/files`);
            setFiles(res.data.files || []);
        } catch (err) {
            console.error("Failed to fetch files", err);
            setError("Unable to retrieve file list. Please ensure the backend is active.");
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-slate-900/40 backdrop-blur-md animate-in fade-in duration-300"
                onClick={onClose}
            ></div>

            {/* Modal Content */}
            <div className="relative w-full max-w-lg bg-white/90 backdrop-blur-2xl rounded-theme border border-white shadow-2xl overflow-hidden animate-in zoom-in-95 duration-300">
                <div className="p-6 border-b border-slate-100 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-primary/10 text-primary rounded-xl">
                            <Hash className="w-5 h-5" />
                        </div>
                        <div>
                            <h3 className="text-lg font-black text-slate-800 tracking-tight">
                                {folderName} <span className="text-slate-400 font-medium">Contents</span>
                            </h3>
                            <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Knowledge Base Index</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-xl hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-all"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="p-6 overflow-y-auto max-h-[60vh] custom-scrollbar">
                    {isLoading ? (
                        <div className="flex flex-col items-center justify-center py-12 gap-3">
                            <Loader2 className="w-8 h-8 text-primary animate-spin" />
                            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Querying Index...</p>
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center py-12 gap-3 text-center">
                            <AlertCircle className="w-8 h-8 text-rose-400" />
                            <p className="text-xs font-bold text-slate-500 max-w-[200px] leading-relaxed italic">{error}</p>
                        </div>
                    ) : files.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-12 gap-3 opacity-40">
                            <FileText className="w-12 h-12 text-slate-300" />
                            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">No Documents Indexed</p>
                        </div>
                    ) : (
                        <div className="grid gap-2">
                            {files.map((file, idx) => (
                                <div
                                    key={idx}
                                    className="flex items-center gap-4 p-4 rounded-2xl bg-white border border-slate-50 hover:border-primary/20 hover:shadow-lg hover:shadow-indigo-100/50 transition-all group"
                                >
                                    <div className="p-2 rounded-xl bg-slate-50 text-slate-400 group-hover:bg-primary/10 group-hover:text-primary transition-all">
                                        <FileText className="w-4 h-4" />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm font-bold text-slate-700 truncate">{file}</p>
                                        <p className="text-[10px] text-emerald-500 font-bold uppercase tracking-wider">Ready for Retrieval</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                <div className="p-6 bg-slate-50/50 border-t border-slate-100">
                    <button
                        onClick={onClose}
                        className="w-full py-4 theme-gradient text-white rounded-2xl text-xs font-black uppercase tracking-widest shadow-lg shadow-primary/20 hover:scale-[1.02] active:scale-95 transition-all"
                    >
                        Close Viewer
                    </button>
                </div>
            </div>
        </div>
    );
};

export default FileViewerModal;
