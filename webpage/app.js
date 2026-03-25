// Application logic for BoneHub public datasets and subjects tables

const NUMERIC_COLUMN_TOKENS = ['id', 'age', 'weight', 'height', 'bmi'];

document.addEventListener('DOMContentLoaded', function() {
    initializeTableController({
        data: typeof datasetsData !== 'undefined' ? datasetsData : [],
        itemLabel: 'dataset',
        itemLabelPlural: 'datasets',
        emptyMessage: 'No datasets found matching your filters.',
        noDataMessage: 'No dataset metadata available.',
        exportFilePrefix: 'bonehub_datasets',
        searchInputId: 'datasetSearchInput',
        clearButtonId: 'clearDatasetFilters',
        toggleColumnsButtonId: 'toggleDatasetColumns',
        exportButtonId: 'exportDatasetsCSV',
        countId: 'datasetCount',
        columnTogglesId: 'datasetColumnToggles',
        columnCheckboxesId: 'datasetColumnCheckboxes',
        tableHeaderId: 'datasetsTableHeader',
        filterRowId: 'datasetsFilterRow',
        tableBodyId: 'datasetsTableBody'
    });

    initializeTableController({
        data: typeof subjectsData !== 'undefined' ? subjectsData : [],
        itemLabel: 'subject',
        itemLabelPlural: 'subjects',
        emptyMessage: 'No subjects found matching your filters.',
        noDataMessage: 'Dataset not loaded. Please run the update script.',
        exportFilePrefix: 'bonehub_subjects',
        searchInputId: 'searchInput',
        clearButtonId: 'clearFilters',
        toggleColumnsButtonId: 'toggleColumns',
        exportButtonId: 'exportCSV',
        countId: 'subjectCount',
        columnTogglesId: 'columnToggles',
        columnCheckboxesId: 'columnCheckboxes',
        tableHeaderId: 'tableHeader',
        filterRowId: 'filterRow',
        tableBodyId: 'tableBody'
    });
});

function initializeTableController(config) {
    const state = {
        ...config,
        allData: Array.isArray(config.data) ? config.data : [],
        filteredData: Array.isArray(config.data) ? [...config.data] : [],
        columnFilters: {},
        visibleColumns: new Set(),
        allColumns: [],
        sortColumn: null,
        sortDirection: 'asc',
        elements: getTableElements(config)
    };

    if (!state.elements.headerRow || !state.elements.filterRow || !state.elements.tbody) {
        return null;
    }

    state.allColumns = state.allData.length > 0 ? Object.keys(state.allData[0]) : [];
    state.visibleColumns = new Set(state.allColumns);

    buildTableStructure(state);
    initializeColumnToggles(state);
    renderTable(state);
    setupEventListeners(state);
    updateStickyColumnOffsets(state);

    return state;
}

function getTableElements(config) {
    return {
        searchInput: document.getElementById(config.searchInputId),
        clearButton: document.getElementById(config.clearButtonId),
        toggleColumnsButton: document.getElementById(config.toggleColumnsButtonId),
        exportButton: document.getElementById(config.exportButtonId),
        countElement: document.getElementById(config.countId),
        columnToggles: document.getElementById(config.columnTogglesId),
        columnCheckboxes: document.getElementById(config.columnCheckboxesId),
        headerRow: document.getElementById(config.tableHeaderId),
        filterRow: document.getElementById(config.filterRowId),
        tbody: document.getElementById(config.tableBodyId)
    };
}

function buildTableStructure(state) {
    const { headerRow, filterRow } = state.elements;

    headerRow.innerHTML = '';
    filterRow.innerHTML = '';

    if (state.allColumns.length === 0) {
        return;
    }

    state.allColumns.forEach((column, index) => {
        const th = document.createElement('th');
        th.textContent = column;
        th.dataset.column = column;
        th.dataset.columnIndex = index;
        th.classList.add('sortable');
        th.addEventListener('click', function() {
            handleSort(state, column);
        });
        headerRow.appendChild(th);

        const td = document.createElement('td');
        td.dataset.column = column;
        td.dataset.columnIndex = index;

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'filter-input';
        input.dataset.column = column;

        if (isNumericColumn(column)) {
            input.placeholder = 'e.g., >100 or <50';
            input.title = 'Supports: >, <, >=, <=, = followed by a number';
        } else {
            input.placeholder = `Filter ${column}...`;
        }

        input.addEventListener('input', function(event) {
            handleColumnFilter(state, column, event.target.value);
        });

        td.appendChild(input);
        filterRow.appendChild(td);
    });

    applyColumnVisibility(state);
    updateSortIndicators(state);
}

function initializeColumnToggles(state) {
    const container = state.elements.columnCheckboxes;
    if (!container) {
        return;
    }

    container.innerHTML = '';

    state.allColumns.forEach(column => {
        const div = document.createElement('div');
        div.className = 'column-checkbox';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `${state.itemLabel}-col-${slugify(column)}`;
        checkbox.checked = state.visibleColumns.has(column);
        checkbox.dataset.column = column;

        const label = document.createElement('label');
        label.htmlFor = checkbox.id;
        label.textContent = column;

        checkbox.addEventListener('change', function() {
            if (this.checked) {
                state.visibleColumns.add(column);
            } else {
                state.visibleColumns.delete(column);
            }
            applyColumnVisibility(state);
        });

        div.appendChild(checkbox);
        div.appendChild(label);
        container.appendChild(div);
    });
}

function setupEventListeners(state) {
    const {
        searchInput,
        clearButton,
        toggleColumnsButton,
        exportButton,
        columnToggles,
        filterRow
    } = state.elements;

    if (searchInput) {
        searchInput.addEventListener('input', function() {
            applyFilters(state);
        });
    }

    if (clearButton) {
        clearButton.addEventListener('click', function() {
            if (searchInput) {
                searchInput.value = '';
            }

            Array.from(filterRow.querySelectorAll('.filter-input')).forEach(input => {
                input.value = '';
            });

            state.columnFilters = {};
            applyFilters(state);
        });
    }

    if (toggleColumnsButton && columnToggles) {
        toggleColumnsButton.addEventListener('click', function() {
            columnToggles.style.display = columnToggles.style.display === 'none' ? 'block' : 'none';
        });
    }

    if (exportButton) {
        exportButton.addEventListener('click', function() {
            exportToCSV(state);
        });
    }

    window.addEventListener('resize', function() {
        updateStickyColumnOffsets(state);
    });
}

function handleSort(state, column) {
    if (state.sortColumn === column) {
        state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        state.sortColumn = column;
        state.sortDirection = 'asc';
    }

    sortFilteredData(state);
    renderTable(state);
}

function handleColumnFilter(state, column, value) {
    if (value.trim() === '') {
        delete state.columnFilters[column];
    } else {
        state.columnFilters[column] = value.trim();
    }

    applyFilters(state);
}

function applyFilters(state) {
    const searchTerm = state.elements.searchInput ? state.elements.searchInput.value.toLowerCase() : '';

    state.filteredData = state.allData.filter(row => {
        for (const [column, filterValue] of Object.entries(state.columnFilters)) {
            const cellValue = formatCellValue(row[column]);

            if (isNumericColumn(column)) {
                const comparison = parseNumericComparison(filterValue);
                if (comparison) {
                    if (!applyNumericFilter(cellValue, comparison)) {
                        return false;
                    }
                    continue;
                }
            }

            if (!cellValue.toLowerCase().includes(filterValue.toLowerCase())) {
                return false;
            }
        }

        if (searchTerm) {
            const rowText = Object.values(row).map(formatCellValue).join(' ').toLowerCase();
            if (!rowText.includes(searchTerm)) {
                return false;
            }
        }

        return true;
    });

    sortFilteredData(state);
    renderTable(state);
}

function sortFilteredData(state) {
    if (!state.sortColumn) {
        return;
    }

    state.filteredData.sort((a, b) => {
        const result = compareValues(a, b, state.sortColumn);
        return state.sortDirection === 'asc' ? result : -result;
    });
}

function renderTable(state) {
    const { tbody } = state.elements;
    tbody.innerHTML = '';

    if (state.allColumns.length === 0) {
        showTableMessage(state, state.noDataMessage);
        updateStats(state);
        return;
    }

    if (state.filteredData.length === 0) {
        showTableMessage(state, state.emptyMessage, state.allColumns.length);
    } else {
        state.filteredData.forEach(row => {
            const tr = document.createElement('tr');

            state.allColumns.forEach((column, index) => {
                const td = document.createElement('td');
                td.dataset.column = column;
                td.dataset.columnIndex = index;

                if (isLinkColumn(column)) {
                    td.innerHTML = createLinkButtons(row[column]);
                } else if (shouldUseExpandableCell(state, column, row[column])) {
                    td.appendChild(createExpandableCellContent(row[column]));
                } else {
                    td.textContent = formatCellValue(row[column]);
                }

                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });
    }

    applyColumnVisibility(state);
    updateSortIndicators(state);
    updateStats(state);
    updateStickyColumnOffsets(state);
}

function applyColumnVisibility(state) {
    const { headerRow, filterRow, tbody } = state.elements;

    Array.from(headerRow.children).forEach((th, index) => {
        toggleColumnVisibility(th, state.visibleColumns.has(state.allColumns[index]));
    });

    Array.from(filterRow.children).forEach((td, index) => {
        toggleColumnVisibility(td, state.visibleColumns.has(state.allColumns[index]));
    });

    Array.from(tbody.children).forEach(tr => {
        Array.from(tr.children).forEach((td, index) => {
            toggleColumnVisibility(td, state.visibleColumns.has(state.allColumns[index]));
        });
    });
}

function toggleColumnVisibility(element, isVisible) {
    if (!element) {
        return;
    }

    if (isVisible) {
        element.classList.remove('column-hidden');
    } else {
        element.classList.add('column-hidden');
    }
}

function updateStickyColumnOffsets(state) {
    const { headerRow, tbody } = state.elements;
    const firstHeaderCell = headerRow ? headerRow.children[0] : null;
    const secondHeaderCell = headerRow ? headerRow.children[1] : null;
    const firstColumnWidth = firstHeaderCell && !firstHeaderCell.classList.contains('column-hidden')
        ? firstHeaderCell.offsetWidth
        : 0;
    const secondColumnWidth = secondHeaderCell && !secondHeaderCell.classList.contains('column-hidden')
        ? secondHeaderCell.offsetWidth
        : 0;

    const table = headerRow ? headerRow.closest('table') : null;
    if (!table) {
        return;
    }

    table.style.setProperty('--sticky-first-column-width', `${firstColumnWidth}px`);
    table.style.setProperty('--sticky-second-column-width', `${secondColumnWidth}px`);

    if (tbody && tbody.children.length > 0) {
        const firstRow = tbody.children[0];
        const firstCell = firstRow.children[0];
        const secondCell = firstRow.children[1];

        if (firstCell && !firstCell.classList.contains('column-hidden')) {
            table.style.setProperty('--sticky-first-column-width', `${firstCell.offsetWidth}px`);
        }

        if (secondCell && !secondCell.classList.contains('column-hidden')) {
            table.style.setProperty('--sticky-second-column-width', `${secondCell.offsetWidth}px`);
        }
    }
}

function updateSortIndicators(state) {
    const { headerRow } = state.elements;

    Array.from(headerRow.children).forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc', 'sort-active');
        if (th.dataset.column === state.sortColumn) {
            th.classList.add('sort-active');
            th.classList.add(state.sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
        }
    });
}

function updateStats(state) {
    const { countElement } = state.elements;
    if (!countElement) {
        return;
    }

    if (state.allColumns.length === 0) {
        countElement.textContent = `No ${state.itemLabelPlural} available`;
        return;
    }

    countElement.textContent = `Showing ${state.filteredData.length} of ${state.allData.length} ${state.itemLabelPlural}`;
}

function showTableMessage(state, message, colspan) {
    const tr = document.createElement('tr');
    const td = document.createElement('td');
    td.className = 'empty-state';
    td.textContent = message;
    td.colSpan = colspan || 1;
    tr.appendChild(td);
    state.elements.tbody.appendChild(tr);
}

function exportToCSV(state) {
    if (state.filteredData.length === 0) {
        alert(`No ${state.itemLabelPlural} to export. Please adjust your filters.`);
        return;
    }

    const columnsToExport = state.allColumns.filter(column => state.visibleColumns.has(column));
    let csvContent = '';

    csvContent += columnsToExport.map(column => `"${column}"`).join(',') + '\n';

    state.filteredData.forEach(row => {
        const rowData = columnsToExport.map(column => {
            let value = formatCellValue(row[column]).replace(/"/g, '""');
            return `"${value}"`;
        });
        csvContent += rowData.join(',') + '\n';
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);

    link.setAttribute('href', url);
    link.setAttribute('download', `${state.exportFilePrefix}_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function normalizeSortValue(value) {
    if (value === null || value === undefined) {
        return '';
    }
    return String(value).trim();
}

function compareValues(a, b, column) {
    const valueA = normalizeSortValue(a[column]);
    const valueB = normalizeSortValue(b[column]);

    if (valueA === '' && valueB === '') {
        return 0;
    }
    if (valueA === '') {
        return 1;
    }
    if (valueB === '') {
        return -1;
    }

    const numA = parseFloat(valueA);
    const numB = parseFloat(valueB);
    const bothNumeric = !Number.isNaN(numA) && !Number.isNaN(numB) && /^-?\d+(\.\d+)?$/.test(valueA) && /^-?\d+(\.\d+)?$/.test(valueB);

    if (bothNumeric) {
        return numA - numB;
    }

    return valueA.localeCompare(valueB, undefined, {
        numeric: true,
        sensitivity: 'base'
    });
}

function formatCellValue(value) {
    return value === null || value === undefined ? '' : String(value);
}

function isLinkColumn(column) {
    return ['Access Link', 'Related Paper', 'URL', 'Paper'].includes(column);
}

function parseNumericComparison(filterValue) {
    const operators = [
        { regex: /^>=\s*(.+)/, op: '>=' },
        { regex: /^<=\s*(.+)/, op: '<=' },
        { regex: /^>\s*(.+)/, op: '>' },
        { regex: /^<\s*(.+)/, op: '<' },
        { regex: /^=\s*(.+)/, op: '=' }
    ];

    for (const { regex, op } of operators) {
        const match = filterValue.match(regex);
        if (match) {
            const value = parseFloat(match[1]);
            if (!Number.isNaN(value)) {
                return { operator: op, value: value };
            }
        }
    }

    return null;
}

function applyNumericFilter(cellValue, comparison) {
    const numericValue = parseFloat(cellValue);
    if (Number.isNaN(numericValue)) {
        return false;
    }

    switch (comparison.operator) {
        case '>':
            return numericValue > comparison.value;
        case '<':
            return numericValue < comparison.value;
        case '>=':
            return numericValue >= comparison.value;
        case '<=':
            return numericValue <= comparison.value;
        case '=':
            return numericValue === comparison.value;
        default:
            return false;
    }
}

function isNumericColumn(columnName) {
    const normalizedColumnWords = columnName
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, ' ')
        .trim()
        .split(/\s+/)
        .filter(Boolean);

    return normalizedColumnWords.some(word => NUMERIC_COLUMN_TOKENS.includes(word));
}

function shouldUseExpandableCell(state, column, value) {
    if (state.itemLabel !== 'subject') {
        return false;
    }

    if (!['Segmentation Available', 'Mesh Available', 'NURBS Available'].includes(column)) {
        return false;
    }

    return getExpandableItems(value).length > 0;
}

function createExpandableCellContent(value) {
    const items = getExpandableItems(value);
    if (items.length === 0) {
        return document.createTextNode('');
    }

    const details = document.createElement('details');
    details.className = 'expandable-cell';

    const summary = document.createElement('summary');
    summary.className = 'expandable-cell-toggle';
    summary.textContent = `Show ${items.length} item${items.length === 1 ? '' : 's'}`;

    const content = document.createElement('div');
    content.className = 'expandable-cell-content';

    const list = document.createElement('ul');
    list.className = 'expandable-cell-list';

    items.forEach(item => {
        const listItem = document.createElement('li');
        listItem.textContent = item;
        list.appendChild(listItem);
    });

    content.appendChild(list);
    details.appendChild(summary);
    details.appendChild(content);

    return details;
}

function getExpandableItems(value) {
    const items = formatCellValue(value)
        .split(';')
        .map(item => item.trim())
        .filter(Boolean);
    
    // Natural sort: numeric-aware, case-insensitive
    items.sort((a, b) => {
        return a.localeCompare(b, undefined, {
            numeric: true,
            sensitivity: 'base'
        });
    });
    
    return items;
}

function createLinkButtons(linkString) {
    if (!linkString || !String(linkString).trim()) {
        return '';
    }

    const links = String(linkString).split(';').map(link => link.trim()).filter(Boolean);
    if (links.length === 0) {
        return '';
    }

    return links.map((link, index) => {
        const buttonText = links.length === 1 ? 'Link' : `Link ${index + 1}`;
        const trackedLink = addBonehubReferral(link);
        return `<a href="${trackedLink}" target="_blank" rel="noopener noreferrer" class="btn-link">${buttonText}</a>`;
    }).join(' ');
}

function addBonehubReferral(link) {
    try {
        const url = new URL(link);
        if (url.protocol === 'http:' || url.protocol === 'https:') {
            if (!url.searchParams.has('utm_source')) {
                url.searchParams.set('utm_source', 'bonehub');
            }
            return url.toString();
        }
    } catch (error) {
        return link;
    }

    return link;
}

function slugify(value) {
    return String(value).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}