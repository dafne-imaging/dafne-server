<?php
declare(strict_types=1);
session_start();

require_once __DIR__ . '/config.php';
require_once __DIR__ . '/lib/db.php';
require_once __DIR__ . '/lib/auth.php';
require_once __DIR__ . '/lib/models.php';

// ---------------------------------------------------------------------------
// CSRF
// ---------------------------------------------------------------------------

if (empty($_SESSION['req_csrf'])) {
    $_SESSION['req_csrf'] = bin2hex(random_bytes(32));
}
$csrf_token = $_SESSION['req_csrf'];

// ---------------------------------------------------------------------------
// reCAPTCHA Enterprise verification
// ---------------------------------------------------------------------------

function verify_recaptcha(string $token): bool
{
    // If no site key is configured, skip verification (development mode).
    if (RECAPTCHA_SITE_KEY === '') {
        return true;
    }
    if ($token === '') {
        return false;
    }

    $url = sprintf(
        'https://recaptchaenterprise.googleapis.com/v1/projects/%s/assessments?key=%s',
        urlencode(RECAPTCHA_PROJECT_ID),
        urlencode(RECAPTCHA_API_KEY)
    );

    $payload = json_encode([
        'event' => [
            'token'          => $token,
            'siteKey'        => RECAPTCHA_SITE_KEY,
            'expectedAction' => 'submit_request',
            'userIpAddress'  => $_SERVER['REMOTE_ADDR'] ?? '',
        ],
    ]);

    $ctx = stream_context_create(['http' => [
        'method'  => 'POST',
        'header'  => "Content-Type: application/json\r\nAccept: application/json",
        'content' => $payload,
        'timeout' => 5,
    ]]);

    $raw = @file_get_contents($url, false, $ctx);
    if ($raw === false) {
        return false;
    }

    $data = json_decode($raw, true) ?? [];
    $props = $data['tokenProperties'] ?? [];
    $risk  = $data['riskAnalysis']    ?? [];

    return ($props['valid']  ?? false) === true
        && ($props['action'] ?? '')    === 'submit_request'
        && ($risk['score']   ?? 0.0)   >= RECAPTCHA_MIN_SCORE;
}

// ---------------------------------------------------------------------------
// Admin notification emails
// ---------------------------------------------------------------------------

function send_request_notification(
    string $name,
    string $email,
    string $reason,
    array  $models,
    string $admin_url
): bool {
    $model_list = empty($models)
        ? '  (none specified)'
        : implode("\n", array_map(fn($m) => "  • {$m}", $models));

    $body = implode("\r\n", [
        'A new user access request has been submitted.',
        '',
        str_repeat('─', 50),
        "Name:    {$name}",
        "Email:   {$email}",
        'Submitted: ' . date('Y-m-d H:i:s') . ' UTC',
        '',
        'Requested model access:',
        $model_list,
        '',
        'Reason for request:',
        $reason,
        str_repeat('─', 50),
        '',
        'Review and approve or reject this request at:',
        $admin_url,
        '',
        'This message was sent automatically by the Dafne server.',
    ]);

    $subject = "New access request from {$name} <{$email}>";
    $headers = implode("\r\n", [
        'From: '        . ADMIN_EMAIL,
        'Reply-To: '    . $email,
        'Content-Type: text/plain; charset=UTF-8',
        'MIME-Version: 1.0',
    ]);

    return mail(ADMIN_EMAIL, $subject, $body, $headers);
}

function send_contact_notification(string $name, string $email, string $message): bool
{
    $body = implode("\r\n", [
        'A contact message has been submitted via the Dafne server.',
        '',
        str_repeat('─', 50),
        "Name:    {$name}",
        "Email:   {$email}",
        'Submitted: ' . date('Y-m-d H:i:s') . ' UTC',
        '',
        'Message:',
        $message,
        str_repeat('─', 50),
        '',
        'This message was sent automatically by the Dafne server.',
    ]);

    $subject = "Message from {$name} <{$email}>";
    $headers = implode("\r\n", [
        'From: '        . ADMIN_EMAIL,
        'Reply-To: '    . $email,
        'Content-Type: text/plain; charset=UTF-8',
        'MIME-Version: 1.0',
    ]);

    return mail(ADMIN_EMAIL, $subject, $body, $headers);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function h(string $s): string
{
    return htmlspecialchars($s, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function db_insert_request(string $name, string $email, string $reason, array $models): int
{
    $s = get_db()->prepare(
        'INSERT INTO user_requests (name, email, reason, requested_models) VALUES (?, ?, ?, ?)'
    );
    $s->execute([$name, $email, $reason, json_encode(array_values($models))]);
    return (int) get_db()->lastInsertId();
}

function base_url(): string
{
    $https = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off');
    $proto = $https ? 'https' : 'http';
    $host  = $_SERVER['HTTP_HOST'] ?? 'localhost';
    $dir   = rtrim(dirname($_SERVER['PHP_SELF']), '/');
    return "{$proto}://{$host}{$dir}";
}

// ---------------------------------------------------------------------------
// Available model types
// ---------------------------------------------------------------------------

$model_types = get_model_types();

// ---------------------------------------------------------------------------
// POST handler
// ---------------------------------------------------------------------------

$success      = false;
$contact_mode = false;
$errors       = [];
$old          = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // CSRF check
    if (!hash_equals($csrf_token, $_POST['csrf_token'] ?? '')) {
        http_response_code(403);
        die('Invalid CSRF token.');
    }

    $form_type = trim($_POST['form_type'] ?? 'request');
    if (!in_array($form_type, ['request', 'contact'], true)) {
        $form_type = 'request';
    }
    $contact_mode = ($form_type === 'contact');

    $name   = trim($_POST['name']   ?? '');
    $email  = trim($_POST['email']  ?? '');
    $reason = trim($_POST['reason'] ?? '');
    $old    = compact('name', 'email', 'reason', 'form_type');

    // Collect and sanitize requested model types (only relevant for API key requests)
    $models = [];
    if (!$contact_mode) {
        $raw_models = (array) ($_POST['models'] ?? []);
        foreach ($raw_models as $m) {
            $clean = sanitize_model_type((string) $m);
            if ($clean !== null && in_array($clean, $model_types, true)) {
                $models[] = $clean;
            }
        }
    }

    // Validate fields
    if ($name === '') {
        $errors[] = 'Please enter your name.';
    }
    if ($email === '' || !filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $errors[] = 'Please enter a valid email address.';
    }
    if ($reason === '') {
        $errors[] = $contact_mode ? 'Please enter your message.' : 'Please describe why you need access.';
    }

    // reCAPTCHA
    if (empty($errors) && !verify_recaptcha($_POST['recaptcha_token'] ?? '')) {
        $errors[] = 'reCAPTCHA verification failed. Please try again.';
    }

    if (empty($errors)) {
        try {
            if ($contact_mode) {
                send_contact_notification($name, $email, $reason);
            } else {
                $request_id = db_insert_request($name, $email, $reason, $models);
                $admin_url  = base_url() . '/admin.php?review=' . $request_id;
                send_request_notification($name, $email, $reason, $models, $admin_url);
            }
            $success = true;
            // Regenerate CSRF token after successful submit to prevent reuse
            $_SESSION['req_csrf'] = bin2hex(random_bytes(32));
        } catch (Throwable $e) {
            $errors[] = 'An error occurred while submitting your request. Please try again later.';
        }
    }
}

?>
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dafne – Contact/Request API Key</title>
<?php if (RECAPTCHA_SITE_KEY !== ''): ?>
<script src="https://www.google.com/recaptcha/enterprise.js?render=<?= h(RECAPTCHA_SITE_KEY) ?>"></script>
<?php endif ?>
<link rel="stylesheet" href="css/request.css">
</head>
<body>

<div class="page-card">
  <div class="page-header">
    <h1>Contact/Request API Key</h1>
    <p>Fill in the form below to send a message or request an account on the Dafne federated learning server.<br>
       An administrator will review your request and contact you by email.</p>
  </div>

  <div class="page-body">

  <?php if ($success): ?>

    <div class="success-icon">✅</div>
    <?php if ($contact_mode): ?>
    <div class="success-title">Message sent!</div>
    <p class="success-text">
      Your message has been forwarded to the administrator.
      They will get back to you at the email address you provided.
    </p>
    <?php else: ?>
    <div class="success-title">Request submitted!</div>
    <p class="success-text">
      Your request has been received. An administrator will review it and send
      your API key to the email address you provided.
    </p>
    <?php endif ?>

  <?php else: ?>

    <?php if (!empty($errors)): ?>
      <div class="flash error">
        <?php if (count($errors) === 1): ?>
          <?= h($errors[0]) ?>
        <?php else: ?>
          Please fix the following:
          <ul><?php foreach ($errors as $e): ?><li><?= h($e) ?></li><?php endforeach ?></ul>
        <?php endif ?>
      </div>
    <?php endif ?>

    <form id="request-form" method="post" novalidate>
      <input type="hidden" name="csrf_token"     value="<?= h($csrf_token) ?>">
      <input type="hidden" name="recaptcha_token" id="recaptcha_token" value="">

      <?php $current_form_type = $old['form_type'] ?? 'request'; ?>
      <div class="form-row form-type-row">
        <label class="radio-option">
          <input type="radio" name="form_type" value="request" id="type-request"
                 <?= $current_form_type === 'request' ? 'checked' : '' ?>>
          Request API Key
        </label>
        <label class="radio-option">
          <input type="radio" name="form_type" value="contact" id="type-contact"
                 <?= $current_form_type === 'contact' ? 'checked' : '' ?>>
          Contact
        </label>
      </div>

      <div class="form-row">
        <label for="name">Full Name</label>
        <input type="text" id="name" name="name" value="<?= h($old['name'] ?? '') ?>"
               autocomplete="name" required>
      </div>

      <div class="form-row">
        <label for="email">Email Address</label>
        <input type="email" id="email" name="email" value="<?= h($old['email'] ?? '') ?>"
               autocomplete="email" required>
      </div>

      <div class="form-row">
        <label for="reason" id="reason-label">Reason for Request</label>
        <textarea id="reason" name="reason" rows="4" required
                  placeholder="Please describe your use case and why you need access to the Dafne server."
        ><?= h($old['reason'] ?? '') ?></textarea>
      </div>

      <?php if (!empty($model_types)): ?>
      <div class="form-row" id="models-row">
        <label>Models Requested</label>
        <div class="models-grid">
          <label class="model-check all-check">
            <input type="checkbox" id="all-models"> All models
          </label>
          <?php foreach ($model_types as $mt): ?>
          <label class="model-check">
            <input type="checkbox" name="models[]" value="<?= h($mt) ?>"
                   class="model-cb"
                   <?= in_array($mt, (array)($_POST['models'] ?? []), true) ? 'checked' : '' ?>>
            <?= h($mt) ?>
          </label>
          <?php endforeach ?>
        </div>
      </div>
      <?php endif ?>

      <button type="submit" class="btn btn-primary" id="submit-btn">Submit Request</button>

      <?php if (RECAPTCHA_SITE_KEY !== ''): ?>
      <p class="recaptcha-note">
        This site is protected by reCAPTCHA.<br>
        The Google <a href="https://policies.google.com/privacy" target="_blank" rel="noopener">Privacy Policy</a>
        and <a href="https://policies.google.com/terms" target="_blank" rel="noopener">Terms of Service</a> apply.
      </p>
      <?php endif ?>
    </form>

  <?php endif ?>

  </div>
</div>

<script>
// Form type toggle (Request API Key / Contact)
const typeRequest = document.getElementById('type-request');
const typeContact = document.getElementById('type-contact');
const modelsRow   = document.getElementById('models-row');
const reasonLabel = document.getElementById('reason-label');
const reasonArea  = document.getElementById('reason');
const submitBtn   = document.getElementById('submit-btn');

function applyFormType(isContact) {
    if (modelsRow) {
        modelsRow.style.display = isContact ? 'none' : '';
    }
    if (isContact) {
        reasonLabel.textContent = 'Message';
        reasonArea.placeholder  = 'Write your message to the administrator.';
        submitBtn.textContent   = 'Send Message';
    } else {
        reasonLabel.textContent = 'Reason for Request';
        reasonArea.placeholder  = 'Please describe your use case and why you need access to the Dafne server.';
        submitBtn.textContent   = 'Submit Request';
    }
}

if (typeRequest && typeContact) {
    // Apply initial state (handles server-side repopulation after error)
    applyFormType(typeContact.checked);

    typeRequest.addEventListener('change', () => applyFormType(false));
    typeContact.addEventListener('change', () => applyFormType(true));
}

// "All models" shortcut
const allCb  = document.getElementById('all-models');
const modelCbs = Array.from(document.querySelectorAll('.model-cb'));

if (allCb) {
    allCb.addEventListener('change', () => {
        modelCbs.forEach(cb => { cb.checked = allCb.checked; });
    });
    modelCbs.forEach(cb => cb.addEventListener('change', () => {
        allCb.checked = modelCbs.length > 0 && modelCbs.every(c => c.checked);
        allCb.indeterminate = !allCb.checked && modelCbs.some(c => c.checked);
    }));
}

<?php if (RECAPTCHA_SITE_KEY !== ''): ?>
// reCAPTCHA v3: obtain a fresh token on submit, then allow the form to post.
const form      = document.getElementById('request-form');
const tokenField = document.getElementById('recaptcha_token');

form.addEventListener('submit', function (e) {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = 'Verifying…';
    grecaptcha.enterprise.ready(function () {
        grecaptcha.enterprise.execute(<?= json_encode(RECAPTCHA_SITE_KEY) ?>, { action: 'submit_request' })
            .then(function (token) {
                tokenField.value = token;
                form.submit();
            })
            .catch(function () {
                submitBtn.disabled = false;
                applyFormType(typeContact && typeContact.checked);
                alert('reCAPTCHA failed to load. Please refresh and try again.');
            });
    });
});
<?php endif ?>
</script>

</body>
</html>