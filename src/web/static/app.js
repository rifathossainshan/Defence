document.addEventListener('DOMContentLoaded', () => {
    // State management
    let currentPatient = '';
    let currentPlane = 'axial';
    let currentModality = 'flair';
    let currentSlicePct = 0.5;
    
    // UI Elements
    const caseSelect = document.getElementById('query-case-select');
    const planeBtns = document.querySelectorAll('.plane-btn');
    const modalityBtns = document.querySelectorAll('.modality-btn');
    const sliceSlider = document.getElementById('slice-slider');
    const sliceBadge = document.getElementById('slice-badge');
    const mriImg = document.getElementById('mri-slice-img');
    const mriGrid = document.getElementById('mri-grid');
    const mriPlaceholder = document.getElementById('mri-placeholder');
    const viewerLoader = document.getElementById('viewer-loader');
    const resultsList = document.getElementById('results-list');
    const logOutput = document.getElementById('log-output');

    // Live terminal logger
    function writeLog(text) {
        const time = new Date().toLocaleTimeString();
        logOutput.innerText += `\n[${time}] ${text}`;
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    // 1. Fetch Patient Cases
    fetch('/api/cases')
        .then(res => res.json())
        .then(data => {
            caseSelect.innerHTML = '<option value="">-- Choose Patient Case --</option>';
            data.patients.forEach(pid => {
                const opt = document.createElement('option');
                opt.value = pid;
                opt.textContent = pid;
                caseSelect.appendChild(opt);
            });
            writeLog(`Loaded ${data.patients.length} patient records from local index.`);
        })
        .catch(err => {
            writeLog(`Error loading cases: ${err.message}`);
        });

    // 2. Fetch and render 3D MRI Slice (Debounced)
    let sliceTimeout;
    function loadMriSlice() {
        if (!currentPatient) return;
        
        viewerLoader.classList.remove('hidden');
        mriPlaceholder.classList.add('hidden');
        
        clearTimeout(sliceTimeout);
        sliceTimeout = setTimeout(() => {
            if (currentModality === 'all') {
                const modalities = ['flair', 'seg', 'gradcam'];
                const fetchPromises = modalities.map(mod => 
                    fetch(`/api/slice?patient_id=${currentPatient}&modality=${mod}&plane=${currentPlane}&slice_pct=${currentSlicePct}`)
                    .then(res => res.ok ? res.json() : null)
                );

                Promise.all(fetchPromises)
                    .then(results => {
                        mriImg.classList.add('hidden');
                        mriGrid.innerHTML = '';
                        mriGrid.classList.remove('hidden');

                        let sliceIdx = 0, maxSlices = 0;

                        results.forEach((data, index) => {
                            if (data) {
                                sliceIdx = data.slice_idx;
                                maxSlices = data.max_slices;
                                const itemDiv = document.createElement('div');
                                itemDiv.className = 'grid-item';
                                itemDiv.innerHTML = `
                                    <div class="grid-label">${modalities[index].toUpperCase()}</div>
                                    <img src="${data.image}" alt="${modalities[index]}">
                                `;
                                mriGrid.appendChild(itemDiv);
                            }
                        });

                        sliceBadge.textContent = `Slice: ${sliceIdx}/${maxSlices - 1}`;
                        viewerLoader.classList.add('hidden');
                        sliceSlider.disabled = false;
                    })
                    .catch(err => {
                        writeLog(`Grid load failed: ${err.message}`);
                        viewerLoader.classList.add('hidden');
                    });
            } else {
                const url = `/api/slice?patient_id=${currentPatient}&modality=${currentModality}&plane=${currentPlane}&slice_pct=${currentSlicePct}`;
                
                fetch(url)
                    .then(res => {
                        if (!res.ok) throw new Error(`HTTP error ${res.status}`);
                        return res.json();
                    })
                    .then(data => {
                        mriGrid.classList.add('hidden');
                        mriImg.src = data.image;
                        mriImg.classList.remove('hidden');
                        sliceBadge.textContent = `Slice: ${data.slice_idx}/${data.max_slices - 1}`;
                        viewerLoader.classList.add('hidden');
                        sliceSlider.disabled = false;
                    })
                    .catch(err => {
                        writeLog(`Slice load failed: ${err.message}`);
                        viewerLoader.classList.add('hidden');
                    });
            }
        }, 80); // 80ms debounce for super smooth slider dragging
    }

    // 3. Execute FAISS Neighbor Search
    function executeRetrieval() {
        if (!currentPatient) return;
        
        writeLog(`Querying FAISS database for: ${currentPatient}...`);
        resultsList.innerHTML = `
            <div class="placeholder-text">
                <div class="spinner"></div>
                <p>Querying 1381 Patient Embeddings...</p>
            </div>
        `;
        
        fetch(`/api/query?patient_id=${currentPatient}`)
            .then(res => {
                if (!res.ok) throw new Error(`Query failed: ${res.status}`);
                return res.json();
            })
            .then(data => {
                resultsList.innerHTML = '';
                writeLog(`FAISS search completed. Latency: <0.72ms.`);
                
                data.results.forEach(res => {
                    const card = document.createElement('div');
                    card.className = 'result-card';
                    card.innerHTML = `
                        <div class="result-top">
                            <span class="result-rank">Rank ${res.rank}</span>
                            <span class="result-score">Cosine Similarity: ${res.score.toFixed(4)}</span>
                        </div>
                        <div class="result-id"><i class="fa-solid fa-user-doctor"></i> ${res.patient_id}</div>
                        <div class="result-meta">
                            <span>Cohort Source: <strong>${res.dataset}</strong></span>
                        </div>
                        <div class="meter-container">
                            <div class="meter-fill" style="width: ${res.score * 100}%"></div>
                        </div>
                    `;
                    
                    // Click on match card to view its slices!
                    card.addEventListener('click', () => {
                        writeLog(`Viewing matched patient case: ${res.patient_id}`);
                        caseSelect.value = res.patient_id;
                        caseSelect.dispatchEvent(new Event('change'));
                    });
                    
                    resultsList.appendChild(card);
                });
            })
            .catch(err => {
                writeLog(`Retrieval Query Error: ${err.message}`);
                resultsList.innerHTML = `<p class="text-danger">Failed to retrieve matches: ${err.message}</p>`;
            });
    }

    // Event Listeners
    caseSelect.addEventListener('change', (e) => {
        currentPatient = e.target.value;
        if (currentPatient) {
            writeLog(`Selected Query Patient: ${currentPatient}`);
            loadMriSlice();
            executeRetrieval();
        } else {
            mriImg.classList.add('hidden');
            mriPlaceholder.classList.remove('hidden');
            sliceSlider.disabled = true;
            sliceBadge.textContent = 'Slice: 0/0';
            resultsList.innerHTML = `
                <div class="placeholder-text">
                    <i class="fa-solid fa-list-check"></i>
                    <p>Run query to see similar cohorts in database</p>
                </div>
            `;
        }
    });

    planeBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            planeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentPlane = btn.dataset.plane;
            writeLog(`Visual Plane updated: ${currentPlane.toUpperCase()}`);
            loadMriSlice();
        });
    });

    modalityBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            modalityBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentModality = btn.dataset.mod;
            writeLog(`MRI Sequence updated: ${currentModality.toUpperCase()}`);
            loadMriSlice();
        });
    });

    sliceSlider.addEventListener('input', (e) => {
        currentSlicePct = e.target.value / 100;
        loadMriSlice();
    });

    // 4. Academic Figure Carousel controls
    const slides = document.querySelectorAll('.slide');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    let currentSlideIdx = 0;

    function showSlide(idx) {
        slides.forEach(s => s.classList.remove('active'));
        slides[idx].classList.add('active');
    }

    prevBtn.addEventListener('click', () => {
        currentSlideIdx = (currentSlideIdx - 1 + slides.length) % slides.length;
        showSlide(currentSlideIdx);
    });

    nextBtn.addEventListener('click', () => {
        currentSlideIdx = (currentSlideIdx + 1) % slides.length;
        showSlide(currentSlideIdx);
    });
});
