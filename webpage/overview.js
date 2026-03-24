// Renders a compact overview dashboard from subjectsData without coupling to app.js.
(function() {
    function parseItemList(value) {
        if (value === null || value === undefined) {
            return [];
        }

        return String(value)
            .split(';')
            .map(item => item.trim())
            .filter(Boolean);
    }

    function normalizeGender(value) {
        const raw = String(value === null || value === undefined ? '' : value).trim().toUpperCase();
        if (raw === 'M' || raw === 'MALE') {
            return 'Male';
        }
        if (raw === 'F' || raw === 'FEMALE') {
            return 'Female';
        }
        if (raw) {
            return raw;
        }
        return 'Unknown';
    }

    function formatNumber(value, digits) {
        return new Intl.NumberFormat('en-US', {
            maximumFractionDigits: typeof digits === 'number' ? digits : 0,
            minimumFractionDigits: typeof digits === 'number' ? digits : 0
        }).format(value);
    }

    function summarizeSubjects(data) {
        const subjects = Array.isArray(data) ? data : [];
        const totalSubjects = subjects.length;
        const datasetIds = new Set();
        const genders = {};
        const ages = [];

        const segmentationItems = [];
        const meshItems = [];
        const nurbsItems = [];

        subjects.forEach(subject => {
            if (subject['Dataset ID'] !== null && subject['Dataset ID'] !== undefined) {
                datasetIds.add(subject['Dataset ID']);
            }

            const gender = normalizeGender(subject['Gender']);
            genders[gender] = (genders[gender] || 0) + 1;

            const age = Number(subject['Age (years)']);
            if (!Number.isNaN(age) && age > 0) {
                ages.push(age);
            }

            segmentationItems.push(...parseItemList(subject['Segmentation Available']));
            meshItems.push(...parseItemList(subject['Mesh Available']));
            nurbsItems.push(...parseItemList(subject['NURBS Available']));
        });

        const allBoneLabels = [...segmentationItems, ...meshItems, ...nurbsItems];
        const uniqueBoneLabels = new Set(allBoneLabels);

        const subjectsWithImage = subjects.filter(subject => subject['Image Available'] === true || subject['Image Available'] === 'true').length;
        const subjectsWithSeg = subjects.filter(subject => parseItemList(subject['Segmentation Available']).length > 0).length;
        const subjectsWithMesh = subjects.filter(subject => parseItemList(subject['Mesh Available']).length > 0).length;
        const subjectsWithNurbs = subjects.filter(subject => parseItemList(subject['NURBS Available']).length > 0).length;

        const boneFrequencies = {};
        allBoneLabels.forEach(label => {
            boneFrequencies[label] = (boneFrequencies[label] || 0) + 1;
        });

        const topBones = Object.entries(boneFrequencies)
            .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
            .slice(0, 8);

        let ageMin = null;
        let ageMax = null;
        let ageMean = null;
        if (ages.length > 0) {
            ageMin = Math.min(...ages);
            ageMax = Math.max(...ages);
            ageMean = ages.reduce((sum, value) => sum + value, 0) / ages.length;
        }

        return {
            totalSubjects,
            totalDatasets: datasetIds.size,
            genders,
            ages,
            ageMin,
            ageMax,
            ageMean,
            ageKnownCount: ages.length,
            uniqueBoneLabelsCount: uniqueBoneLabels.size,
            subjectsWithImage,
            totalSegmentationMasks: segmentationItems.length,
            totalMeshes: meshItems.length,
            totalNurbs: nurbsItems.length,
            subjectsWithSeg,
            subjectsWithMesh,
            subjectsWithNurbs,
            topBones
        };
    }

    function createMetricCard(label, value, detail) {
        const card = document.createElement('article');
        card.className = 'overview-card';

        const labelEl = document.createElement('p');
        labelEl.className = 'overview-card-label';
        labelEl.textContent = label;

        const valueEl = document.createElement('p');
        valueEl.className = 'overview-card-value';
        valueEl.textContent = value;

        const detailEl = document.createElement('p');
        detailEl.className = 'overview-card-detail';
        detailEl.textContent = detail;

        card.appendChild(labelEl);
        card.appendChild(valueEl);
        card.appendChild(detailEl);

        return card;
    }

    function createBarRow(label, count, total, colorClass) {
        const row = document.createElement('div');
        row.className = 'overview-bar-row';

        const head = document.createElement('div');
        head.className = 'overview-bar-head';
        head.innerHTML = `<span>${label}</span><span>${count}</span>`;

        const track = document.createElement('div');
        track.className = 'overview-bar-track';

        const fill = document.createElement('div');
        fill.className = `overview-bar-fill ${colorClass || ''}`.trim();
        const ratio = total > 0 ? Math.max(0, Math.min(100, (count / total) * 100)) : 0;
        fill.style.width = `${ratio.toFixed(1)}%`;
        fill.setAttribute('aria-hidden', 'true');

        track.appendChild(fill);
        row.appendChild(head);
        row.appendChild(track);

        return row;
    }

    function renderOverview() {
        const metricsRoot = document.getElementById('overviewMetrics');
        const genderBarsRoot = document.getElementById('overviewGenderBars');
        const assetBarsRoot = document.getElementById('overviewAssetBars');
        const ageStatsRoot = document.getElementById('overviewAgeStats');
        const topBonesRoot = document.getElementById('overviewTopBones');

        if (!metricsRoot || !genderBarsRoot || !assetBarsRoot || !ageStatsRoot || !topBonesRoot) {
            return;
        }

        if (typeof subjectsData === 'undefined' || !Array.isArray(subjectsData) || subjectsData.length === 0) {
            metricsRoot.innerHTML = '<p class="overview-empty">No subject data available to summarize.</p>';
            return;
        }

        const summary = summarizeSubjects(subjectsData);

        metricsRoot.innerHTML = '';
        metricsRoot.appendChild(createMetricCard(
            'Subjects',
            formatNumber(summary.totalSubjects),
            `${formatNumber(summary.totalDatasets)} contributing datasets`
        ));
        metricsRoot.appendChild(createMetricCard(
            'Unique Bone Labels',
            formatNumber(summary.uniqueBoneLabelsCount),
            'Across segmentation, mesh, and NURBS fields'
        ));
        metricsRoot.appendChild(createMetricCard(
            'Segmentation Masks',
            formatNumber(summary.totalSegmentationMasks),
            `${formatNumber(summary.subjectsWithSeg)} subjects with at least one segmentation`
        ));
        metricsRoot.appendChild(createMetricCard(
            'Meshes',
            formatNumber(summary.totalMeshes),
            `${formatNumber(summary.subjectsWithMesh)} subjects with at least one mesh`
        ));
        metricsRoot.appendChild(createMetricCard(
            'NURBS',
            formatNumber(summary.totalNurbs),
            `${formatNumber(summary.subjectsWithNurbs)} subjects with at least one NURBS label`
        ));

        genderBarsRoot.innerHTML = '';
        const orderedGenders = Object.entries(summary.genders).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
        if (orderedGenders.length === 0) {
            genderBarsRoot.innerHTML = '<p class="overview-empty">No gender metadata available.</p>';
        } else {
            orderedGenders.forEach(([gender, count], index) => {
                const styleClass = index % 2 === 0 ? 'fill-primary' : 'fill-secondary';
                genderBarsRoot.appendChild(createBarRow(gender, count, summary.totalSubjects, styleClass));
            });
        }

        assetBarsRoot.innerHTML = '';
        const assets = [
            { label: 'Image', count: summary.subjectsWithImage, className: 'fill-image' },
            { label: 'Segmentation', count: summary.subjectsWithSeg, className: 'fill-primary' },
            { label: 'Mesh', count: summary.subjectsWithMesh, className: 'fill-secondary' },
            { label: 'NURBS', count: summary.subjectsWithNurbs, className: 'fill-accent' }
        ];
        assets.forEach(asset => {
            assetBarsRoot.appendChild(createBarRow(asset.label, asset.count, summary.totalSubjects, asset.className));
        });

        ageStatsRoot.innerHTML = '';
        const ageItems = [
            ['Known ages', `${formatNumber(summary.ageKnownCount)} / ${formatNumber(summary.totalSubjects)}`],
            ['Average age', summary.ageMean === null ? 'N/A' : `${formatNumber(summary.ageMean, 1)} years`],
            ['Age range', summary.ageMin === null || summary.ageMax === null ? 'N/A' : `${formatNumber(summary.ageMin)} - ${formatNumber(summary.ageMax)} years`]
        ];
        ageItems.forEach(([label, value]) => {
            const row = document.createElement('div');
            row.className = 'overview-list-row';
            row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
            ageStatsRoot.appendChild(row);
        });

        topBonesRoot.innerHTML = '';
        if (summary.topBones.length === 0) {
            topBonesRoot.innerHTML = '<li class="overview-empty">No bone labels available.</li>';
        } else {
            summary.topBones.forEach(([label, count]) => {
                const li = document.createElement('li');
                li.innerHTML = `<span>${label}</span><strong>${formatNumber(count)}</strong>`;
                topBonesRoot.appendChild(li);
            });
        }
    }

    document.addEventListener('DOMContentLoaded', renderOverview);
})();
