// theme-engine.js

// ==========================================
// 1. DARK MODE LOGIC
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    const body = document.body;
    const toggleBtn = document.getElementById('theme-toggle');
    const themeText = document.getElementById('theme-text');
    const themeIcon = document.getElementById('theme-icon');

    const moonIcon = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
    const sunIcon = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';

    // Apply saved theme immediately on load
    if (localStorage.getItem('theme') === 'dark') {
        body.classList.add('dark-mode');
        if (themeText) themeText.textContent = 'Light Mode';
        if (themeIcon) themeIcon.innerHTML = sunIcon;
    }

    // Handle button clicks (if the button exists on the page)
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            body.classList.toggle('dark-mode');
            const isDark = body.classList.contains('dark-mode');

            // Save preference to browser memory
            localStorage.setItem('theme', isDark ? 'dark' : 'light');

            // Swap Icon and Text
            if (themeText) themeText.textContent = isDark ? 'Light Mode' : 'Dark Mode';
            if (themeIcon) themeIcon.innerHTML = isDark ? sunIcon : moonIcon;
        });
    }
});

// ==========================================
// 2. AUTO-INJECT SYNTAX HIGHLIGHTING (Prism.js)
// ==========================================
// This saves you from copying <link> and <script> tags into every HTML file!

// Inject the CSS Theme
const prismCSS = document.createElement('link');
prismCSS.rel = 'stylesheet';
prismCSS.href = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css';
document.head.appendChild(prismCSS);

// Inject the Javascript Engines (in order)
const scriptsToLoad = [
    'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-yaml.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js'
];

scriptsToLoad.forEach(src => {
    const script = document.createElement('script');
    script.src = src;
    script.async = false; // Forces them to load sequentially
    document.body.appendChild(script);
});