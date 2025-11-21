// register.js - Cleaned and API-Ready Version

// 1. DOM Elements (Correctly scoped at the top)
const registerForm = document.getElementById('registerForm');
const errorMessage = document.getElementById('errorMessage');
const first_nameInput = document.getElementById('first_name');
const last_nameInput = document.getElementById('last_name');
const emailInput = document.getElementById('email');
const passwordInput = document.getElementById('password');
const confirmPasswordInput = document.getElementById('confirmPassword');

// Initialize
function init() {
     registerForm.addEventListener('submit', handleSubmit);
    confirmPasswordInput.addEventListener('input', validatePasswordMatch);
    
    const inputs = [first_nameInput, last_nameInput, emailInput, passwordInput, confirmPasswordInput];
    inputs.forEach(input => {
        input.addEventListener('input', clearError);
    });
}

async function handleSubmit(e) {
    e.preventDefault();

    const formData = new FormData(registerForm);

    try {
        const response = await fetch('/auth/register', {
            method: 'POST',
            body: formData   // ← NO headers, NO JSON
        });

        if (response.redirected) {
            window.location.href = response.url;
        } else {
            const error = await response.json();
            showError(error.detail || "Registration failed.");
        }
        
    } catch (error) {
        showError("Network error.");
    }
}


// 3. Validate form (Uses locally defined variables)
function validateForm(first_name, last_name, email, password, confirmPassword) {
    if (!first_name || !last_name || !email || !password || !confirmPassword) {
        showError('Please fill in all fields');
        return false;
    }

    if (!isValidEmail(email)) {
        showError('Please enter a valid email address');
        return false;
    }

    if (password.length < 8) {
        showError('Password must be at least 8 characters long');
        return false;
    }

    if (password !== confirmPassword) {
        showError('Passwords do not match');
        return false;
    }
    
    // NOTE: The user existence check against localStorage is removed, 
    // as FastAPI handles this against the real database.

    return true;
}

// Email validation
function isValidEmail(email) {
    return email.includes('@') && email.includes('.');
}

// Real-time password match validation
function validatePasswordMatch() {
    const password = passwordInput.value;
    const confirmPassword = confirmPasswordInput.value;
    
    if (confirmPassword && password !== confirmPassword) {
        confirmPasswordInput.style.borderColor = 'rgba(244, 67, 54, 0.6)';
    } else {
        confirmPasswordInput.style.borderColor = 'rgba(129, 199, 132, 0.3)';
    }
}

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('show');
    
    errorMessage.style.animation = 'none';
    setTimeout(() => {
        errorMessage.style.animation = 'shake 0.5s ease';
    }, 10);
}

// Clear error message
function clearError() {
    errorMessage.classList.remove('show');
}

// Initialize the form
init();