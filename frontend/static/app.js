const { createApp, ref, onMounted } = Vue;

if (document.getElementById('app')) {
    createApp({
        setup() {
            const transactions = ref([]);
            const metrics = ref(null);
            const loading = ref(true);
            const recovering = ref(null);
            let chartInstance = null;

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
                        plugins: {
                            legend: { display: false },
                            title: { display: true, text: 'Agent Actions', font: {size: 10}, padding: 0 }
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

            return { transactions, metrics, loading, recovering, fetchData, runRecovery };
        }
    }).mount('#app');
}

if (document.getElementById('audit-app')) {
    createApp({
        setup() {
            const logs = ref([]);
            const loading = ref(true);

            const fetchLogs = async () => {
                loading.value = true;
                try {
                    const res = await fetch(`/api/payments/audit-logs/${window.txnId}`);
                    logs.value = await res.json();
                    // Sort descending
                    logs.value.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
                } catch (e) {
                    console.error(e);
                } finally {
                    loading.value = false;
                }
            };

            onMounted(fetchLogs);

            return { logs, loading, fetchLogs };
        }
    }).mount('#audit-app');
}
