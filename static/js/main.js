// Simple page intro animations
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.querySelector('.btn-neon');
  if (btn) {
    btn.addEventListener('mouseenter', () => {
      btn.style.transform = 'scale(1.1)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.transform = 'scale(1)';
    });
  }
});
