document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('transaction-form');
    const submitBtn = document.getElementById('submit-btn');
    const spinner = document.getElementById('spinner');
    
    // UI Elements
    const emptyState = document.getElementById('empty-state');
    const verdictContainer = document.getElementById('verdict-container');
    const statusHalo = document.getElementById('status-halo');
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    
    // Metrics Elements
    const probVal = document.getElementById('prob-val');
    const latencyVal = document.getElementById('latency-val');
    
    // API Endpoint
    const API_URL = '/predict';

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // 1. Enter Loading State
        submitBtn.querySelector('span').classList.add('hide');
        spinner.classList.remove('hide');
        submitBtn.disabled = true;
        
        emptyState.classList.add('hide');
        verdictContainer.classList.add('hide');

        // 2. Extract specific form values
        const distVal = document.getElementById('dist_to_merch').value;
        const payload = {
            user_id: document.getElementById('user_id').value,
            merchant_id: document.getElementById('merchant_id').value,
            amount: parseFloat(document.getElementById('amount').value),
            category: document.getElementById('category').value,
            dist_to_merch: distVal ? parseFloat(distVal) : -1.0
        };

        try {
            // 3. Call the FastAPI Engine
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();

            // 4. Parse Results & Update UI
            updateVerdictUI(data);

        } catch (error) {
            console.error('Error hitting fraud API:', error);
            alert('Failed to connect to the Fraud Detection API. Ensure the FastAPI server is running with CORS enabled.');
        } finally {
            // 5. Reset Button
            submitBtn.querySelector('span').classList.remove('hide');
            spinner.classList.add('hide');
            submitBtn.disabled = false;
        }
    });

    function updateVerdictUI(data) {
        // Reset classes
        statusHalo.className = 'status-halo';
        statusText.className = 'status-text';
        
        if (data.is_fraud) {
            statusHalo.classList.add('flagged');
            statusIcon.textContent = '✕';
            statusText.textContent = 'FRAUD BLOCKED';
            statusText.classList.add('text-flagged');
        } else {
            statusHalo.classList.add('authorize');
            statusIcon.textContent = '✓';
            statusText.textContent = 'AUTHORIZED';
            statusText.classList.add('text-authorize');
        }

        // Apply animated numbers
        const probPct = (data.fraud_probability * 100).toFixed(2);
        probVal.textContent = `${probPct}%`;
        
        // Change color based on severity
        if (probPct > 50) probVal.style.color = '#ef4444';      // Red
        else if (probPct > 15) probVal.style.color = '#facc15'; // Yellow
        else probVal.style.color = '#10b981';                   // Green

        latencyVal.textContent = `${data.latency_ms} ms`;
        
        // Unhide card with a small delay for dramatic effect
        setTimeout(() => {
            verdictContainer.classList.remove('hide');
        }, 100);
    }
});
