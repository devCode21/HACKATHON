
  document.querySelector('form').addEventListener('submit', function (e) {
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm-password').value;

    if (password !== confirmPassword) {
      e.preventDefault(); // Prevent form submission
      alert('Passwords do not match. Please try again.');
    }
  });

  document.getElementById('email').addEventListener('input', function () {
    const email = this.value;
    const emailPattern = /^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}$/;
    if (!emailPattern.test(email)) {
      this.style.borderColor = 'red';
    } else {
      this.style.borderColor = 'green';
    }
  });

  const passwordInput = document.getElementById('password');
  const toggleButton = document.getElementById('toggle-password');

  toggleButton.addEventListener('click', () => {
    if (passwordInput.type === 'password') {
      passwordInput.type = 'text';
      toggleButton.textContent = 'Hide';
    } else {
      passwordInput.type = 'password';
      toggleButton.textContent = 'Show';
    }
  });

