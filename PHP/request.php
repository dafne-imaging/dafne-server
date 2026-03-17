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
// reCAPTCHA v3 verification
// ---------------------------------------------------------------------------

function verify_recaptcha(string $token): bool
{
    // If no secret key is configured, skip verification (development mode).
    if (RECAPTCHA_SECRET_KEY === '') {
        return true;
    }
    if ($token === '') {
        return false;
    }
    $ctx  = stream_context_create(['http' => [
        'method'  => 'POST',
        'header'  => 'Content-Type: application/x-www-form-urlencoded',
        'content' => http_build_query([
            'secret'   => RECAPTCHA_SECRET_KEY,
            'response' => $token,
            'remoteip' => $_SERVER['REMOTE_ADDR'] ?? '',
        ]),
        'timeout' => 5,
    ]]);
    $raw  = @file_get_contents('https://www.google.com/recaptcha/api/siteverify', false, $ctx);
    if ($raw === false) {
        return false;
    }
    $data = json_decode($raw, true) ?? [];
    return ($data['success'] ?? false) === true
        && ($data['action']  ?? '') === 'submit_request'
        && ($data['score']   ?? 0.0) >= RECAPTCHA_MIN_SCORE;
}

// ---------------------------------------------------------------------------
// Admin notification email
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

$success    = false;
$errors     = [];
$old        = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // CSRF check
    if (!hash_equals($csrf_token, $_POST['csrf_token'] ?? '')) {
        http_response_code(403);
        die('Invalid CSRF token.');
    }

    $name   = trim($_POST['name']   ?? '');
    $email  = trim($_POST['email']  ?? '');
    $reason = trim($_POST['reason'] ?? '');
    $old    = compact('name', 'email', 'reason');

    // Collect and sanitize requested model types
    $raw_models = (array) ($_POST['models'] ?? []);
    $models = [];
    foreach ($raw_models as $m) {
        $clean = sanitize_model_type((string) $m);
        if ($clean !== null && in_array($clean, $model_types, true)) {
            $models[] = $clean;
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
        $errors[] = 'Please describe why you need access.';
    }

    // reCAPTCHA
    if (empty($errors) && !verify_recaptcha($_POST['recaptcha_token'] ?? '')) {
        $errors[] = 'reCAPTCHA verification failed. Please try again.';
    }

    if (empty($errors)) {
        try {
            $request_id = db_insert_request($name, $email, $reason, $models);
            $admin_url  = base_url() . '/admin.php?review=' . $request_id;
            send_request_notification($name, $email, $reason, $models, $admin_url);
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
<title>Dafne – Request Access</title>
<?php if (RECAPTCHA_SITE_KEY !== ''): ?>
<script src="https://www.google.com/recaptcha/api.js?render=<?= h(RECAPTCHA_SITE_KEY) ?>"></script>
<?php endif ?>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 14px;
    background: #f0f2f5;
    color: #1a1a2e;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px 60px;
}

.page-card {
    background: #fff;
    border-radius: 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,.1);
    width: 100%;
    max-width: 560px;
    overflow: hidden;
}

.page-header {
    background: #1a3a5c;
    color: #fff;
    padding: 24px 28px;
}
.page-header h1 { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
.page-header p  { opacity: .75; font-size: 13px; line-height: 1.5; }

.page-body { padding: 28px; }

/* Flash / errors */
.flash { border-radius: 6px; padding: 12px 14px; margin-bottom: 18px; font-size: 13px; }
.flash.error   { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }
.flash.success { background: #ecfdf5; border: 1px solid #6ee7b7; color: #065f46; }
.flash ul { padding-left: 18px; margin-top: 4px; }

/* Form */
.form-row { margin-bottom: 18px; }
.form-row label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: #475569;
    margin-bottom: 5px;
    text-transform: uppercase;
    letter-spacing: .04em;
}
.form-row input[type="text"],
.form-row input[type="email"],
.form-row textarea {
    width: 100%;
    padding: 9px 12px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    font-size: 14px;
    font-family: inherit;
    color: #1a1a2e;
    outline: none;
    transition: border-color .15s;
    resize: vertical;
}
.form-row input:focus,
.form-row textarea:focus { border-color: #1a3a5c; }

/* Models */
.models-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 8px;
    margin-top: 8px;
}
.model-check {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 10px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    cursor: pointer;
    transition: background .12s, border-color .12s;
    font-size: 13px;
    user-select: none;
}
.model-check:hover { background: #f8fafc; border-color: #94a3b8; }
.model-check input { accent-color: #1a3a5c; flex-shrink: 0; }
.model-check.all-check {
    border-color: #bfdbfe;
    background: #eff6ff;
    color: #1d4ed8;
    font-weight: 600;
}
.model-check.all-check:hover { background: #dbeafe; }

/* Submit */
.btn {
    display: inline-block;
    padding: 10px 22px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: opacity .15s;
}
.btn:hover:not(:disabled) { opacity: .85; }
.btn:disabled { opacity: .5; cursor: not-allowed; }
.btn-primary { background: #1a3a5c; color: #fff; width: 100%; }

/* reCAPTCHA badge note */
.recaptcha-note {
    margin-top: 14px;
    font-size: 11px;
    color: #94a3b8;
    text-align: center;
    line-height: 1.5;
}
.recaptcha-note a { color: #64748b; }

/* Success state */
.success-icon { font-size: 48px; text-align: center; margin-bottom: 16px; }
.success-title { font-size: 18px; font-weight: 700; color: #065f46; text-align: center; margin-bottom: 10px; }
.success-text  { font-size: 14px; color: #475569; text-align: center; line-height: 1.6; }
</style>
</head>
<body>

<div class="page-card">
  <div class="page-header">
    <h1>Request Access</h1>
    <p>Fill in the form below to request an account on the Dafne federated learning server.<br>
       An administrator will review your request and contact you by email.</p>
  </div>

  <div class="page-body">

  <?php if ($success): ?>

    <div class="success-icon">✅</div>
    <div class="success-title">Request submitted!</div>
    <p class="success-text">
      Your request has been received. An administrator will review it and send
      your API key to the email address you provided.
    </p>

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
        <label for="reason">Reason for Request</label>
        <textarea id="reason" name="reason" rows="4" required
                  placeholder="Please describe your use case and why you need access to the Dafne server."
        ><?= h($old['reason'] ?? '') ?></textarea>
      </div>

      <?php if (!empty($model_types)): ?>
      <div class="form-row">
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
const submitBtn = document.getElementById('submit-btn');

form.addEventListener('submit', function (e) {
    e.preventDefault();
    submitBtn.disabled = true;
    submitBtn.textContent = 'Verifying…';
    grecaptcha.ready(function () {
        grecaptcha.execute(<?= json_encode(RECAPTCHA_SITE_KEY) ?>, { action: 'submit_request' })
            .then(function (token) {
                tokenField.value = token;
                form.submit();
            })
            .catch(function () {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit Request';
                alert('reCAPTCHA failed to load. Please refresh and try again.');
            });
    });
});
<?php endif ?>
</script>

</body>
</html>