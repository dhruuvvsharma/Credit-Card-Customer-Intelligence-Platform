document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predict-form");
    if (!form) return;

    form.addEventListener("submit", () => {
        const btn = form.querySelector(".submit-btn");
        if (btn) {
            btn.disabled = true;
            btn.textContent = "Running assessment…";
        }
    });
});