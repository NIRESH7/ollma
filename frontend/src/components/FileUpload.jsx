import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';

const FileUpload = ({ onUploadSuccess, activeFolder }) => {
    const [status, setStatus] = useState('idle'); // idle, uploading, success, error, partial
    const [message, setMessage] = useState('');
    const [details, setDetails] = useState([]);
    const [progress, setProgress] = useState(0);
    const [ocrProgress, setOcrProgress] = useState(null); // { current, total, percent }
    const pollIntervalRef = useRef(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        };
    }, []);

    const handleFileChange = async (e) => {
        if (e.target.files && e.target.files.length > 0) {
            const selectedFiles = Array.from(e.target.files);
            const jobId = `session_job_${Date.now()}`;
            setStatus('uploading');
            setMessage(`Uploading ${selectedFiles.length} file(s)...`);
            setProgress(0);
            setOcrProgress(null);
            setDetails([]);

            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });
            formData.append('folder', activeFolder || "default");
            formData.append('job_id', jobId);

            // Start polling for OCR progress
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            let pollCount = 0;
            pollIntervalRef.current = setInterval(async () => {
                pollCount++;
                if (pollCount > 300) { // Safety: 5 minutes max
                    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
                    return;
                }
                try {
                    const res = await axios.get(`http://127.0.0.1:8000/upload-status/${jobId}`);
                    if (res.data && res.data.status === 'processing') {
                        setOcrProgress({
                            current: res.data.current_page,
                            total: res.data.total_pages,
                            percent: res.data.progress
                        });
                        setMessage(`OCR Processing: Page ${res.data.current_page}/${res.data.total_pages} (${res.data.progress}%)`);
                    } else if (res.data.status === 'completed' || res.data.error) {
                        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
                    }
                } catch (err) {
                    console.error("Polling error:", err);
                    // On network error, don't necessarily stop, but eventually the safety count will kill it
                }
            }, 1000);

            try {
                const response = await axios.post('http://127.0.0.1:8000/upload/', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    onUploadProgress: (progressEvent) => {
                        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                        setProgress(percentCompleted);
                        if (percentCompleted === 100) {
                            setMessage("Upload complete. Waiting for OCR extraction...");
                        }
                    }
                });

                const results = response.data.results;
                const failures = results.filter(r => r.status === 'failed' || r.status === 'error');
                const successes = results.filter(r => r.status === 'success');

                if (failures.length === 0) {
                    setStatus('success');
                    setMessage(`Successfully uploaded ${successes.length} file(s) to "${activeFolder || "default"}"`);
                } else if (successes.length === 0) {
                    setStatus('error');
                    setMessage(`Failed to upload any files. Check errors below.`);
                } else {
                    setStatus('partial');
                    setMessage(`Uploaded ${successes.length} file(s), but ${failures.length} failed.`);
                }

                setDetails(results);

                if (successes.length > 0 && onUploadSuccess) onUploadSuccess();

            } catch (error) {
                console.error(error);
                setStatus('error');
                const errorMsg = error.response?.data?.detail || error.message;
                setMessage(`Upload failed: ${errorMsg}`);
            } finally {
                if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
            }
        }
    };

    return (
        <div className="relative group">
            <input
                type="file"
                multiple
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                disabled={status === 'uploading'}
            />

            <div className={`p-4 rounded-[20px] border transition-all duration-500 flex flex-col items-center justify-center gap-3 ${status === 'uploading'
                ? 'bg-white/60 border-indigo-200'
                : 'bg-white/40 border-white/80 hover:border-white hover:bg-white/60 shadow-sm'
                }`}>

                <div className={`p-2.5 rounded-xl transition-all duration-300 ${status === 'uploading' ? 'bg-indigo-600 text-white animate-pulse' : 'bg-white/80 text-indigo-500 shadow-sm border border-white/60'}`}>
                    {status === 'uploading' ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                        <Upload className="w-4 h-4" />
                    )}
                </div>

                <div className="text-center">
                    <p className={`text-[10px] font-black uppercase tracking-[0.2em] ${status === 'uploading' ? 'text-indigo-600' : 'text-slate-800'}`}>
                        {status === 'uploading' ? 'Ingestion Active' : 'Import Documents'}
                    </p>
                    <p className="text-[9px] text-slate-500 mt-0.5 font-bold tracking-[0.1em] opacity-80 uppercase">
                        {status === 'uploading' ? 'Neural Ingestion Active' : 'Neural Ingestion Protocol'}
                    </p>
                </div>

                {status === 'uploading' && (
                    <div className="w-full max-w-[120px] h-1.5 bg-white/20 rounded-full overflow-hidden mt-1 p-[1px] border border-white/40">
                        <div
                            className="h-full bg-indigo-500 rounded-full transition-all duration-500 shadow-[0_0_10px_rgba(99,102,241,0.4)]"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                )}
            </div>

            {/* Status Feedback Overlay - Translucent Floating Card */}
            {(status === 'success' || status === 'error' || status === 'partial') && (
                <div className="mt-4 p-4 rounded-[20px] glass-panel animate-fade-in z-30">
                    <div className="flex items-center gap-3 mb-3">
                        <div className={`p-2 rounded-lg ${status === 'success' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'}`}>
                            {status === 'success' ? <CheckCircle className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />}
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-[10px] font-black text-slate-800 uppercase tracking-[0.2em]">
                                {status === 'success' ? 'Sync Complete' : 'Network Alert'}
                            </p>
                            <p className="text-[9px] text-slate-600 font-bold truncate">{message}</p>
                        </div>
                    </div>

                    {details.length > 0 && (
                        <div className="space-y-1.5 max-h-32 overflow-y-auto custom-scrollbar pr-2 border-t border-white/5 pt-3">
                            {details.map((res, idx) => (
                                <div key={idx} className="flex items-center gap-2 text-[9px] py-1.5 px-2.5 rounded-lg bg-white/5 border border-white/5">
                                    <div className={`w-1.5 h-1.5 rounded-full ${res.status === 'success' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]'}`}></div>
                                    <span className="text-slate-700 font-bold truncate flex-1">{res.filename}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    <button
                        onClick={() => setStatus('idle')}
                        className="w-full mt-4 py-2.5 bg-indigo-600/10 hover:bg-indigo-600/20 text-indigo-700 text-[10px] font-black uppercase tracking-[0.2em] rounded-lg transition-all border border-indigo-600/20 active:scale-95"
                    >
                        Acknowledge
                    </button>
                </div>
            )}
        </div>



    );
};

export default FileUpload;
