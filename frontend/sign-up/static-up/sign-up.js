document.addEventListener('DOMContentLoaded', () => {

  // Login elements
  const usernameInputLogin = document.getElementById('usernameInput');
  const passwordInputLogin = document.getElementById('passwordInput');
  const togglePasswordBtn = document.getElementById('togglePassword');
  const demoAccountButtons = document.querySelectorAll('.demo-account');
const loginForm = document.getElementById('loginForm');
const loginMessage = document.getElementById('loginMessage'); // Assuming this is where 'Invalid credentials' appears

// --- NEW FIX: Clear error on submit ---
if (loginForm && loginMessage) {
    loginForm.addEventListener('submit', () => {
        // Clear the error message immediately upon pressing 'Log in'
        loginMessage.textContent = ''; 
    });
}
  // Demo account autofill
  if (demoAccountButtons && demoAccountButtons.length) {
      demoAccountButtons.forEach(btn => {
          btn.addEventListener('click', () => {
              usernameInputLogin.value = btn.dataset.user;
              passwordInputLogin.value = btn.dataset.pass;
          });
      });
  }

  // Toggle password
  if (togglePasswordBtn && passwordInputLogin) {
      togglePasswordBtn.addEventListener('click', () => {
          if (passwordInputLogin.type === 'password') {
              passwordInputLogin.type = 'text';
              togglePasswordBtn.textContent = '🙈';
          } else {
              passwordInputLogin.type = 'password';
              togglePasswordBtn.textContent = '👁';
          }
      });
  }

  // Chat app initialization
  function initChatElements() { /* ... */ }
  function init() { /* ... */ }

});