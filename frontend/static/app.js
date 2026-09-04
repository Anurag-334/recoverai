const { createApp, ref, computed, onMounted } = Vue;

if (document.getElementById('app')) {
    createApp({
        setup() {
            const transactions = ref([]);
            const metrics = ref(null);
            const loading = ref(true);
            const recovering = ref(null);
            let chartInstance = null;

            // Agent Actions Breakdown Modal State
            const showModal = ref(false);
            const activeFilter = ref('all');
            const breakdownData = ref({ counts: { all: 0, restored: 0, negotiating: 0, reminded: 0, escalated: 0 }, actions: [] });
            const loadingBreakdown = ref(false);

            const fetchBreakdown = async () => {
                loadingBreakdown.value = true;
                try {
                    const res = await fetch('/api/payments/agent-actions-breakdown');
                    if (res.ok) {
                        breakdownData.value = await res.json();
                    }
                } catch (e) {
                    console.error("Error fetching breakdown:", e);
                } finally {
                    loadingBreakdown.value = false;
                }
            };

            const openBreakdownModal = (filter = 'all') => {
                activeFilter.value = filter;
                showModal.value = true;
                fetchBreakdown();
            };

            const closeModal = () => {
                showModal.value = false;
            };

            const filteredActions = computed(() => {
                if (!breakdownData.value || !breakdownData.value.actions) return [];
                if (activeFilter.value === 'all') return breakdownData.value.actions;
                return breakdownData.value.actions.filter(a => a.category === activeFilter.value);
            });

            const fetchData = async () => {
                loading.value = true;
                try {
                    const [txRes, metricsRes] = await Promise.all([
                        fetch('/api/payments/transactions'),
                        fetch('/api/payments/metrics')
                    ]);
                    transactions.value = await txRes.json();
                    metrics.value = await metricsRes.json();
                    renderChart();
                } catch (e) {
                    console.error(e);
                } finally {
                    loading.value = false;
                }
            };

            const renderChart = () => {
                if (!metrics.value || !metrics.value.actions) return;
                const ctx = document.getElementById('actionsChart');
                if (!ctx) return;
                
                const labels = Object.keys(metrics.value.actions);
                const data = Object.values(metrics.value.actions);
                
                if (chartInstance) chartInstance.destroy();
                
                chartInstance = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: labels,
                        datasets: [{
                            data: data,
                            backgroundColor: ['#4F46E5', '#10B981', '#F59E0B', '#EF4444', '#6B7280']
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        onClick: (event, elements) => {
                            if (elements && elements.length > 0) {
                                const index = elements[0].index;
                                const actionKey = labels[index];
                                let cat = 'all';
                                if (actionKey === 'retry_payment') cat = 'restored';
                                else if (actionKey === 'negotiation_reply') cat = 'negotiating';
                                else if (actionKey.includes('reminder')) cat = 'reminded';
                                else if (actionKey === 'escalate_to_merchant') cat = 'escalated';
                                openBreakdownModal(cat);
                            } else {
                                openBreakdownModal('all');
                            }
                        },
                        plugins: {
                            legend: { display: false },
                            title: { display: true, text: 'Agent Actions (Click)', font: {size: 10, weight: 'bold'}, padding: 0 }
                        }
                    }
                });
            };

            const runRecovery = async (transaction_id) => {
                recovering.value = transaction_id;
                try {
                    await fetch(`/api/recovery/evaluate/${transaction_id}`, { method: 'POST' });
                    await fetchData();
                } catch (e) {
                    console.error(e);
                    alert("Error running recovery.");
                } finally {
                    recovering.value = null;
                }
            };

            onMounted(fetchData);

            return { 
                transactions, 
                metrics, 
                loading, 
                recovering, 
                fetchData, 
                runRecovery,
                showModal,
                activeFilter,
                breakdownData,
                loadingBreakdown,
                openBreakdownModal,
                closeModal,
                filteredActions
            };
        }
    }).mount('#app');
}

if (document.getElementById('audit-app')) {
    createApp({
        setup() {
            const logs = ref([]);
            const loading = ref(true);
            
            const customerReply = ref("");
            const sendingReply = ref(false);
            const replyStatus = ref("");

            const fetchLogs = async (silent = false) => {
                if (!silent) loading.value = true;
                try {
                    const res = await fetch(`/api/payments/audit-logs/${window.txnId}`);
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    const data = await res.json();
                    if (Array.isArray(data)) {
                        data.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                        logs.value = data;
                    }
                } catch (e) {
                    console.error("Failed to load audit logs:", e);
                } finally {
                    loading.value = false;
                }
            };
            
            const sendReply = async () => {
                const message = customerReply.value.trim();
                if (!message) return;
                
                sendingReply.value = true;
                replyStatus.value = "";
                try {
                    const res = await fetch(`/api/webhooks/whatsapp-reply/${window.txnId}`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ message: message })
                    });
                    
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    
                    const data = await res.json();
                    replyStatus.value = data.reply || "Response received.";
                    customerReply.value = "";
                    await fetchLogs(true); // Silent reload so page does not flash loading spinner
                } catch (e) {
                    console.error("Failed to send simulation reply:", e);
                    replyStatus.value = "Error sending reply. Check console for details.";
                } finally {
                    sendingReply.value = false;
                }
            };

            onMounted(() => fetchLogs(false));

            return { logs, loading, fetchLogs, customerReply, sendingReply, replyStatus, sendReply };
        }
    }).mount('#audit-app');
}
