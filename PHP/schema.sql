-- Dafne server – MySQL schema
-- Run once to initialise the database:
--   mysql -u root -p dafne < schema.sql

CREATE TABLE IF NOT EXISTS users (
    id           INT          NOT NULL AUTO_INCREMENT,
    name         VARCHAR(255) NOT NULL,
    email        VARCHAR(255) NOT NULL,
    -- Unsalted SHA-256 hex digest of the 16-character random API key.
    api_key_hash CHAR(64)     NOT NULL,
    -- Administration permission: allows changing permissions for other users.
    is_admin     TINYINT(1)   NOT NULL DEFAULT 0,
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_users_api_key_hash (api_key_hash)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Per-model access permissions (read canonical models, upload trained models).
-- A user may only interact with model types listed here.
CREATE TABLE IF NOT EXISTS users_accesspermissions (
    user_id    INT          NOT NULL,
    model_type VARCHAR(100) NOT NULL,

    PRIMARY KEY (user_id, model_type),
    CONSTRAINT fk_access_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Per-model merge permissions (list/download uploads, push merged models).
-- A user may only perform merge-client operations on model types listed here.
CREATE TABLE IF NOT EXISTS users_mergepermissions (
    user_id    INT          NOT NULL,
    model_type VARCHAR(100) NOT NULL,

    PRIMARY KEY (user_id, model_type),
    CONSTRAINT fk_merge_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Pending user access requests submitted via the public request form.
-- Rows are created with status='pending' and updated to 'approved' or 'rejected'
-- by an admin. Approved rows produce a corresponding row in `users`.
CREATE TABLE IF NOT EXISTS user_requests (
    id               INT          NOT NULL AUTO_INCREMENT,
    name             VARCHAR(255) NOT NULL,
    email            VARCHAR(255) NOT NULL,
    reason           TEXT         NOT NULL,
    -- JSON array of model type strings the user requested access to, e.g. '["Thigh","Hip"]'.
    requested_models TEXT         NOT NULL,
    status           ENUM('pending','approved','rejected') NOT NULL DEFAULT 'pending',
    created_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    reviewed_at      DATETIME     NULL,
    -- Admin user who approved or rejected the request.
    reviewed_by      INT          NULL,

    PRIMARY KEY (id),
    CONSTRAINT fk_request_reviewer
        FOREIGN KEY (reviewed_by) REFERENCES users (id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Server-side activity log written by server_log().
CREATE TABLE IF NOT EXISTS server_log (
    id         INT          NOT NULL AUTO_INCREMENT,
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    message    TEXT         NOT NULL,

    PRIMARY KEY (id),
    KEY idx_server_log_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;