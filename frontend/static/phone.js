const { createApp, ref, onMounted, computed, watch, nextTick, onUnmounted } = Vue;

// ─── Web Audio Sound Engine ───────────────────────────────────────────
class SoundEngine {
    constructor() {
        this.ctx = null;
        this.enabled = true;
        this.ringtoneInterval = null;
    }

    init() {
        if (!this.ctx) {
            this.ctx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    playRingtone() {
        if (!this.enabled) return () => {};
        this.init();
        let playing = true;

        const playRing = () => {
            if (!playing) return;
            const osc1 = this.ctx.createOscillator();
            const osc2 = this.ctx.createOscillator();
            const gain = this.ctx.createGain();

            osc1.type = 'sine';
            osc1.frequency.value = 440;
            osc2.type = 'sine';
            osc2.frequency.value = 480;

            osc1.connect(gain);
            osc2.connect(gain);
            gain.connect(this.ctx.destination);

            gain.gain.setValueAtTime(0, this.ctx.currentTime);
            gain.gain.linearRampToValueAtTime(0.3, this.ctx.currentTime + 0.1);
            gain.gain.setValueAtTime(0.3, this.ctx.currentTime + 1.9);
            gain.gain.linearRampToValueAtTime(0, this.ctx.currentTime + 2.0);

            osc1.start(this.ctx.currentTime);
            osc2.start(this.ctx.currentTime);
            osc1.stop(this.ctx.currentTime + 2.0);
            osc2.stop(this.ctx.currentTime + 2.0);
        };

        playRing();
        this.ringtoneInterval = setInterval(playRing, 4000);

        return () => {
            playing = false;
            if (this.ringtoneInterval) clearInterval(this.ringtoneInterval);
        };
    }

    playConnectTone() {
        if (!this.enabled) return;
        this.init();
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.frequency.setValueAtTime(800, this.ctx.currentTime);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        gain.gain.setValueAtTime(0.1, this.ctx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.2);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.2);
    }

    playHangupTone() {
        if (!this.enabled) return;
        this.init();
        const freqs = [800, 600, 400];
        freqs.forEach((freq, i) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime + (i * 0.15));
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            gain.gain.setValueAtTime(0, this.ctx.currentTime + (i * 0.15));
            gain.gain.linearRampToValueAtTime(0.1, this.ctx.currentTime + (i * 0.15) + 0.02);
            gain.gain.linearRampToValueAtTime(0, this.ctx.currentTime + (i * 0.15) + 0.13);
            osc.start(this.ctx.currentTime + (i * 0.15));
            osc.stop(this.ctx.currentTime + (i * 0.15) + 0.15);
        });
    }

    playNotificationPing() {
        if (!this.enabled) return;
        this.init();
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(1200, this.ctx.currentTime);
        osc.connect(gain);
        gain.connect(this.ctx.destination);
        gain.gain.setValueAtTime(0, this.ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0.2, this.ctx.currentTime + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.01, this.ctx.currentTime + 0.5);
        osc.start();
        osc.stop(this.ctx.currentTime + 0.5);
    }
}

// ─── Phone App ────────────────────────────────────────────────────────
if (document.getElementById('phone-app')) {
    createApp({
        setup() {
            // ── State ──────────────────────────────────────────────
            const callState = ref('idle');
            const transactions = ref([]);
            const selectedTxnId = ref('');
            const selectedTxn = ref(null);
            const transcript = ref([]);
            const callTimer = ref(0);
            const isListening = ref(false);
            const isMuted = ref(false);
            const isSpeaking = ref(false);
            const isProcessing = ref(false);
            const suggestedResponses = ref([]);
            const callSummary = ref(null);
            const notification = ref(null);
            const showNotification = ref(false);
            const customerInput = ref('');
            const showTextInput = ref(false);
            const soundEnabled = ref(true);

            const currentTime = ref('');
            const currentDate = ref('');
            const transcriptContainer = ref(null);
            const panelTranscriptContainer = ref(null);

            let timerInterval = null;
            let ringtoneStopFn = null;
            const soundEngine = new SoundEngine();

            let recognition = null;
            let synthesis = window.speechSynthesis;

            // ── Computed ───────────────────────────────────────────
            const formattedTimer = computed(() => {
                const m = Math.floor(callTimer.value / 60).toString().padStart(2, '0');
                const s = (callTimer.value % 60).toString().padStart(2, '0');
                return `${m}:${s}`;
            });

            // ── Clock ──────────────────────────────────────────────
            const updateTime = () => {
                const now = new Date();
                currentTime.value = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                currentDate.value = now.toLocaleDateString([], { weekday: 'long', month: 'long', day: 'numeric' });
            };

            // ── Speech Recognition ─────────────────────────────────
            const initSpeechRecognition = () => {
                const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
                if (SpeechRec) {
                    recognition = new SpeechRec();
                    recognition.continuous = false;
                    recognition.interimResults = true;
                    recognition.lang = 'en-IN';

                    recognition.onstart = () => {
                        isListening.value = true;
                    };

                    recognition.onresult = (event) => {
                        let finalTranscript = '';
                        for (let i = event.resultIndex; i < event.results.length; ++i) {
                            if (event.results[i].isFinal) {
                                finalTranscript += event.results[i][0].transcript;
                            }
                        }
                        if (finalTranscript) {
                            sendUserMessage(finalTranscript);
                        }
                    };

                    recognition.onerror = (event) => {
                        console.error('Speech recognition error', event.error);
                        isListening.value = false;
                    };

                    recognition.onend = () => {
                        isListening.value = false;
                        if (callState.value === 'active' && !isMuted.value && !isProcessing.value && !isSpeaking.value) {
                            try { recognition.start(); } catch(e) {}
                        }
                    };
                } else {
                    // No Speech API — force text input mode
                    showTextInput.value = true;
                }
            };

            // ── Text-to-Speech ─────────────────────────────────────
            const speak = (text) => {
                if (!synthesis) return;
                synthesis.cancel();

                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                utterance.lang = 'en-IN';

                const voices = synthesis.getVoices();
                const preferredVoice = voices.find(v =>
                    v.name.includes('Female') || (v.lang.includes('en') && v.name.includes('Google'))
                ) || voices.find(v => v.lang.startsWith('en')) || voices[0];
                if (preferredVoice) utterance.voice = preferredVoice;

                utterance.onstart = () => {
                    isSpeaking.value = true;
                    if (recognition && isListening.value) {
                        recognition.stop();
                    }
                };

                utterance.onend = () => {
                    isSpeaking.value = false;
                    if (recognition && callState.value === 'active' && !isMuted.value) {
                        try { recognition.start(); } catch(e) {}
                    }
                };

                utterance.onerror = () => {
                    isSpeaking.value = false;
                };

                synthesis.speak(utterance);
            };

            let currentAudio = null;
            const playBase64Audio = (base64) => {
                if (currentAudio) {
                    currentAudio.pause();
                }
                if (recognition && isListening.value) recognition.stop();
                
                currentAudio = new Audio("data:audio/mp3;base64," + base64);
                currentAudio.onplay = () => { isSpeaking.value = true; };
                currentAudio.onended = () => {
                    isSpeaking.value = false;
                    if (callState.value === 'active' && !isMuted.value) {
                        try { recognition.start(); } catch(e) {}
                    }
                };
                currentAudio.onerror = () => { isSpeaking.value = false; };
                currentAudio.play().catch(e => {
                    console.error("Audio play error", e);
                    isSpeaking.value = false;
                });
            };

            // ── API Integration ────────────────────────────────────
            const fetchTransactions = async () => {
                try {
                    const res = await fetch('/api/voice/transactions');
                    if (res.ok) {
                        const data = await res.json();
                        // API returns a flat array of transactions
                        transactions.value = Array.isArray(data) ? data : (data.transactions || []);
                    }
                } catch (err) {
                    console.error('Failed to fetch transactions:', err);
                }

                // Pre-select transaction from query param
                if (window.preselectedTxnId && window.preselectedTxnId !== '') {
                    selectedTxnId.value = window.preselectedTxnId;
                    selectTransaction();
                }
            };

            const selectTransaction = () => {
                selectedTxn.value = transactions.value.find(t => t.transaction_id === selectedTxnId.value) || null;
                transcript.value = [];
                suggestedResponses.value = [];
            };

            // ── Helpers ────────────────────────────────────────────
            const scrollToBottom = async () => {
                await nextTick();
                if (transcriptContainer.value) {
                    transcriptContainer.value.scrollTop = transcriptContainer.value.scrollHeight;
                }
                if (panelTranscriptContainer.value) {
                    panelTranscriptContainer.value.scrollTop = panelTranscriptContainer.value.scrollHeight;
                }
            };

            const addMessage = (role, content) => {
                transcript.value.push({
                    role,
                    content,
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                });
                scrollToBottom();
            };

            const triggerNotification = (notif) => {
                if (!notif) return;
                // Map notification type to icon
                const iconMap = {
                    'payment_link': '🔗',
                    'split_payment': '💳',
                    'promise_logged': '📝'
                };
                notification.value = {
                    icon: iconMap[notif.type] || '🔔',
                    title: notif.title,
                    body: notif.body
                };
                showNotification.value = true;
                soundEngine.playNotificationPing();
                setTimeout(() => { showNotification.value = false; }, 4000);
            };

            // ── Call Flow ──────────────────────────────────────────
            const startRinging = () => {
                soundEngine.enabled = soundEnabled.value;
                soundEngine.init();
                callState.value = 'ringing';
                ringtoneStopFn = soundEngine.playRingtone();
            };

            const acceptCall = async () => {
                if (ringtoneStopFn) ringtoneStopFn();
                soundEngine.playConnectTone();
                callState.value = 'active';
                callTimer.value = 0;
                transcript.value = [];
                timerInterval = setInterval(() => callTimer.value++, 1000);

                // Call initiation API
                isProcessing.value = true;
                try {
                    const res = await fetch(`/api/voice/initiate/${selectedTxnId.value}`, { method: 'POST' });
                    if (res.ok) {
                        const data = await res.json();
                        const greeting = data.greeting;
                        isProcessing.value = false;

                        addMessage('agent', greeting.spoken_reply);
                        if (greeting.audio_base64) {
                            playBase64Audio(greeting.audio_base64);
                        } else {
                            speak(greeting.spoken_reply);
                        }

                        // Set suggested responses from API
                        if (greeting.suggested_responses && greeting.suggested_responses.length > 0) {
                            suggestedResponses.value = greeting.suggested_responses;
                        }

                        // Handle notification if present
                        if (greeting.notification) {
                            triggerNotification(greeting.notification);
                        }
                    } else {
                        throw new Error('API error');
                    }
                } catch (err) {
                    console.error('Initiate call error:', err);
                    isProcessing.value = false;
                    const fallbackGreeting = `Hello, this is RecoverAI calling regarding your pending payment of ₹${selectedTxn.value?.amount || 0}. Is this a good time to talk?`;
                    addMessage('agent', fallbackGreeting);
                    speak(fallbackGreeting);
                    suggestedResponses.value = ["Yes, tell me more", "Why did it fail?", "I'll pay later", "Can I pay half?"];
                }

                // Start listening after greeting
                if (recognition && !isMuted.value) {
                    setTimeout(() => {
                        try { recognition.start(); } catch(e) {}
                    }, 500);
                }
            };

            const sendUserMessage = async (msgText = null) => {
                const text = msgText || customerInput.value.trim();
                if (!text || isProcessing.value) return;

                customerInput.value = '';
                addMessage('customer', text);

                // Stop listening/speaking while processing
                if (recognition && isListening.value) {
                    recognition.stop();
                }
                if (synthesis) synthesis.cancel();
                isSpeaking.value = false;

                isProcessing.value = true;

                try {
                    const res = await fetch(`/api/voice/turn/${selectedTxnId.value}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text })
                    });

                    if (res.ok) {
                        const data = await res.json();
                        const reply = data.reply;

                        isProcessing.value = false;
                        addMessage('agent', reply.spoken_reply);
                        if (reply.audio_base64) {
                            playBase64Audio(reply.audio_base64);
                        } else {
                            speak(reply.spoken_reply);
                        }

                        // Update suggested responses
                        if (reply.suggested_responses && reply.suggested_responses.length > 0) {
                            suggestedResponses.value = reply.suggested_responses;
                        }

                        // Handle notification (payment link, split, promise)
                        if (reply.notification) {
                            triggerNotification(reply.notification);
                        }
                    } else {
                        throw new Error('API error');
                    }
                } catch (err) {
                    console.error('Turn error:', err);
                    isProcessing.value = false;
                    const fallback = "I understand. Would you like me to send you a payment link so you can complete this at your convenience?";
                    addMessage('agent', fallback);
                    speak(fallback);
                    suggestedResponses.value = ["Yes, send link", "I'll pay later", "Can I split the payment?"];
                }
            };

            const endCall = async () => {
                if (ringtoneStopFn) ringtoneStopFn();
                if (timerInterval) clearInterval(timerInterval);
                if (recognition) { try { recognition.stop(); } catch(e) {} }
                if (synthesis) synthesis.cancel();
                if (currentAudio) currentAudio.pause();
                isSpeaking.value = false;
                isListening.value = false;

                soundEngine.playHangupTone();
                callState.value = 'ended';

                try {
                    const res = await fetch(`/api/voice/end/${selectedTxnId.value}`, { method: 'POST' });
                    if (res.ok) {
                        const data = await res.json();
                        const summary = data.summary;
                        callSummary.value = {
                            duration: formattedTimer.value,
                            turns: summary.duration_turns,
                            intent: summary.final_intent,
                            status: summary.final_intent === 'unknown' ? 'Call Completed' : `Action: ${summary.final_intent.replace(/_/g, ' ')}`
                        };
                    } else {
                        throw new Error('API error');
                    }
                } catch (e) {
                    console.error('End call error:', e);
                    callSummary.value = {
                        duration: formattedTimer.value,
                        turns: Math.floor(transcript.value.length / 2),
                        intent: 'completed',
                        status: 'Call Completed'
                    };
                }
            };

            const resetCall = () => {
                if (currentAudio) currentAudio.pause();
                callState.value = 'idle';
                transcript.value = [];
                callTimer.value = 0;
                showTextInput.value = false;
                callSummary.value = null;
                suggestedResponses.value = [];
                notification.value = null;
                showNotification.value = false;
            };

            const toggleMute = () => {
                isMuted.value = !isMuted.value;
                if (isMuted.value && recognition) {
                    try { recognition.stop(); } catch(e) {}
                } else if (!isMuted.value && callState.value === 'active' && recognition && !isSpeaking.value) {
                    try { recognition.start(); } catch(e) {}
                }
            };

            const toggleListening = () => {
                if (!recognition) {
                    // No speech API, toggle text input instead
                    showTextInput.value = !showTextInput.value;
                    return;
                }
                if (isListening.value) {
                    recognition.stop();
                } else {
                    try { recognition.start(); } catch(e) {}
                }
            };

            // ── Lifecycle ──────────────────────────────────────────
            let clockInterval = null;

            onMounted(() => {
                updateTime();
                clockInterval = setInterval(updateTime, 1000);
                fetchTransactions();
                initSpeechRecognition();

                // Pre-load voices
                if (window.speechSynthesis) {
                    window.speechSynthesis.onvoiceschanged = () => {
                        window.speechSynthesis.getVoices();
                    };
                }
            });

            onUnmounted(() => {
                if (clockInterval) clearInterval(clockInterval);
                if (timerInterval) clearInterval(timerInterval);
                if (ringtoneStopFn) ringtoneStopFn();
                if (recognition) { try { recognition.stop(); } catch(e) {} }
                if (synthesis) synthesis.cancel();
                if (currentAudio) currentAudio.pause();
            });

            watch(soundEnabled, (newVal) => {
                soundEngine.enabled = newVal;
            });

            // ── Return ─────────────────────────────────────────────
            return {
                callState, transactions, selectedTxnId, selectedTxn,
                transcript, callTimer, formattedTimer,
                isListening, isMuted, isSpeaking, isProcessing,
                suggestedResponses, callSummary,
                notification, showNotification,
                customerInput, showTextInput, soundEnabled,
                currentTime, currentDate,
                transcriptContainer, panelTranscriptContainer,
                selectTransaction, startRinging, acceptCall,
                endCall, resetCall, sendUserMessage,
                toggleMute, toggleListening
            };
        }
    }).mount('#phone-app');
}
