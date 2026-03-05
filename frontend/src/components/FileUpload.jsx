import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2, Table2 } from 'lucide-react';

const FileUpload = ({ onUploadSuccess, activeFolder }) => {
    const [status, setStatus] = useState('idle'); // idle, uploading, success, error, partial
    const [message, setMessage] = useState('');
    const [details, setDetails] = useState([]);
    const [progress, setProgress] = useState(0);
    const [ocrProgress, setOcrProgress] = useState(null); // { current, total, percent }
    const [isExcelUpload, setIsExcelUpload] = useState(false);
    const [sheetInfo, setSheetInfo] = useState([]); // For Excel sheet metadata
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
            const isExcel = selectedFiles.some(f => /\.(xlsx|xls)$/i.test(f.name));
            setIsExcelUpload(isExcel);
            setStatus('uploading');
            setMessage(`Uploading ${selectedFiles.length} file(s)...`);
            setProgress(0);
            setOcrProgress(null);
            setSheetInfo([]);
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
                    const res = await axios.get(`http://127.0.0.1:8001/upload-status/${jobId}`);
                    if (res.data && res.data.status === 'processing') {
                        setOcrProgress({
                            current: res.data.current_page,
                            total: res.data.total_pages,
                            percent: res.data.progress
                        });
                        setMessage(isExcelUpload
                            ? `Processing Excel: ${res.data.progress}% complete...`
                            : `OCR Processing: Page ${res.data.current_page}/${res.data.total_pages} (${res.data.progress}%)`);
                    } else if (res.data.status === 'completed' || res.data.error) {
                        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
                    }
                } catch (err) {
                    console.error("Polling error:", err);
                }
            }, 1000);

            try {
                const response = await axios.post('http://127.0.0.1:8001/upload/', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    onUploadProgress: (progressEvent) => {
                        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                        setProgress(percentCompleted);
                        if (percentCompleted === 100) {
                            setMessage("Upload complete. Processing file...");
                        }
                    }
                });

                const results = response.data.results;
                const failures = results.filter(r => r.status === 'failed' || r.status === 'error');
                const successes = results.filter(r => r.status === 'success');

                // Extract Excel sheet info from successful results
                const allSheetInfo = [];
                successes.forEach(r => {
                    if (r.details?.sheet_info && r.details.sheet_info.length > 0) {
                        allSheetInfo.push({
                            file: r.filename,
                            sheets: r.details.sheet_info,
                            total_rows: r.details.total_rows,
                            sheets_count: r.details.sheets_count,
                        });
                    }
                });
                setSheetInfo(allSheetInfo);

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
                accept=".pdf,.docx,.txt,.xlsx,.xls"
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
                        {status === 'uploading' ? 'Neural Ingestion Active' : 'PDF · DOCX · TXT · XLSX · XLS'}
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

                    {/* Excel Sheet Info Panel */}
                    {sheetInfo.length > 0 && (
                        <div className="mb-3 space-y-2 border-t border-white/10 pt-3">
                            {sheetInfo.map((info, idx) => (
                                <div key={idx} className="p-2.5 rounded-xl bg-indigo-500/5 border border-indigo-500/15">
                                    <div className="flex items-center gap-2 mb-1.5">
                                        <Table2 className="w-3 h-3 text-indigo-500" />
                                        <p className="text-[9px] font-black text-indigo-700 uppercase tracking-widest truncate">{info.file}</p>
                                    </div>
                                    <p className="text-[9px] text-slate-500 font-bold mb-1">
                                        {info.sheets_count} sheet{info.sheets_count > 1 ? 's' : ''} · {info.total_rows?.toLocaleString()} rows ingested
                                    </p>
                                    <div className="flex flex-wrap gap-1 mt-1">
                                        {info.sheets.map((s, si) => (
                                            <span key={si} className="px-2 py-0.5 rounded-full bg-indigo-100/60 text-indigo-700 text-[8px] font-black border border-indigo-200/50">
                                                {s.sheet} ({s.rows} rows)
                                            </span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {details.length > 0 && (
                        <div className="space-y-1.5 max-h-32 overflow-y-auto custom-scrollbar pr-2 border-t border-white/5 pt-3">
                            {details.map((res, idx) => (
                                <div key={idx} className="flex items-center gap-2 text-[9px] py-1.5 px-2.5 rounded-lg bg-white/5 border border-white/5">
                                    <div className={`w-1.5 h-1.5 rounded-full ${res.status === 'success' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]'}`}></div>
                                    <span className="text-slate-700 font-bold truncate flex-1">{res.filename}</span>
                                    {res.details?.status === 'Ingested (Excel)' && (
                                        <span className="text-indigo-500 font-black text-[8px] uppercase">XLSX</span>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    <button
                        onClick={() => { setStatus('idle'); setSheetInfo([]); }}
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
