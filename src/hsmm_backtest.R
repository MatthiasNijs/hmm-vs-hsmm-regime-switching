# Expanding-window HSMM regime history for SPY.
#
# 2-state Gaussian HSMM with shifted-Poisson sojourn times (mhsmm). The embedded
# transition matrix has a zero diagonal, so all regime persistence comes from the
# sojourn distributions. The model is re-estimated every 21 trading days on all
# data up to the previous day; each day's regime is the filtered state at t-1.
#
# Run from the repository root:  Rscript src/hsmm_backtest.R
# Reads data/spy_features.csv, writes results/hsmm_regimes.csv.

library(mhsmm)

set.seed(547)

# ---------- Helpers ----------

make_hsmm_data <- function(mat) {
  x <- list(x = mat, N = nrow(mat))
  class(x) <- "hsmm.data"
  x
}

build_start_model <- function(X_scaled, J = 2) {
  # K-means init for multivariate Gaussian emissions
  km <- kmeans(X_scaled, centers = J, nstart = 10)

  mu_list <- lapply(1:J, function(j) {
    colMeans(X_scaled[km$cluster == j, , drop = FALSE])
  })
  sigma_list <- lapply(1:J, function(j) {
    cov(X_scaled[km$cluster == j, , drop = FALSE])
  })

  parms_emission <- list(mu = mu_list, sigma = sigma_list)

  # Initial state distribution: uniform
  init <- rep(1 / J, J)

  # Embedded Markov chain: no self transitions (durations via sojourn)
  P <- matrix(c(0, 1,
                1, 0),
              nrow = J, byrow = TRUE)

  # Shifted Poisson durations (initial guesses; EM re-estimates)
  sojourn_start <- list(
    lambda = c(30, 10),
    shift  = c(20, 20),
    type   = "poisson"
  )

  hsmmspec(
    init           = init,
    transition     = P,
    parms.emission = parms_emission,
    sojourn        = sojourn_start,
    dens.emission  = dmvnorm.hsmm
  )
}

# ---------- 1) Load features ----------

df <- read.csv("data/spy_features.csv")
df$date <- as.Date(df$date)
df <- df[order(df$date), ]
rownames(df) <- NULL

feature_cols <- c("ret_1d",
                  "ret_20d",
                  "park_vol_20d",
                  "body_range_10d",
                  "flip_rate_10d")

df <- df[complete.cases(df[, feature_cols]), ]
rownames(df) <- NULL

dates_full <- df$date
X_full     <- as.matrix(df[, feature_cols])
n_total    <- nrow(df)

# ---------- 2) Backtest configuration ----------

train_start_date <- as.Date("2002-01-01")
first_pred_date  <- as.Date("2006-01-03")  # first allocation date
retrain_every    <- 21                     # retrain every 21 trading days
J                <- 2                      # number of HSMM states
M_max            <- 250                    # max duration for EM & predict

train_start_idx <- which(dates_full >= train_start_date)[1]
first_pred_idx  <- which(dates_full >= first_pred_date)[1]
if (is.na(train_start_idx) || is.na(first_pred_idx) || first_pred_idx - 1 < train_start_idx) {
  stop("Not enough data before first_pred_date.")
}

cat("Initial training window:",
    as.character(dates_full[train_start_idx]), "to",
    as.character(dates_full[first_pred_idx - 1]), "\n")

# ---------- 3) Expanding-window HSMM backtest ----------

results <- vector("list", n_total - first_pred_idx + 1)

current_fit         <- NULL
current_scaling     <- NULL
current_trend_state <- NULL
current_chop_state  <- NULL
current_seq_scaled  <- NULL
last_seq_idx_global <- NA_integer_

for (i in seq(first_pred_idx, n_total)) {

  # --- 3a) Retrain at the first date, then every 'retrain_every' steps ---
  if (is.null(current_fit) || ((i - first_pred_idx) %% retrain_every) == 0) {
    train_end_idx <- i - 1
    train_idx     <- train_start_idx:train_end_idx

    cat("Retraining at", as.character(dates_full[i]),
        "on data up to", as.character(dates_full[train_end_idx]), "\n")

    X_train <- X_full[train_idx, , drop = FALSE]

    # Scale features using only the training window
    train_mean <- colMeans(X_train)
    train_sd   <- apply(X_train, 2, sd)
    train_sd[train_sd == 0] <- 1

    X_train_scaled <- scale(X_train, center = train_mean, scale = train_sd)

    fit <- hsmmfit(
      x               = make_hsmm_data(X_train_scaled),
      model           = build_start_model(X_train_scaled, J = J),
      mstep           = mstep.mvnorm,
      M               = M_max,
      maxit           = 200,
      lock.transition = FALSE,
      lock.d          = FALSE,
      graphical       = FALSE
    )

    # Label the state with the higher mean ret_1d on the training window as "trend"
    pred_train  <- predict(fit, make_hsmm_data(X_train_scaled), method = "smoothed", M = M_max)
    state_means <- tapply(df$ret_1d[train_idx], pred_train$s, mean, na.rm = TRUE)
    trend_state <- as.integer(names(which.max(state_means)))

    current_fit         <- fit
    current_scaling     <- list(mean = train_mean, sd = train_sd)
    current_trend_state <- trend_state
    current_chop_state  <- setdiff(1:J, trend_state)
    current_seq_scaled  <- X_train_scaled
    last_seq_idx_global <- train_end_idx
  }

  # --- 3b) Extend the scaled sequence up to (i - 1) ---
  if ((i - 1) > last_seq_idx_global) {
    extra_idx <- (last_seq_idx_global + 1):(i - 1)
    X_extra   <- X_full[extra_idx, , drop = FALSE]

    X_extra_scaled <- sweep(X_extra, 2, current_scaling$mean, "-")
    X_extra_scaled <- sweep(X_extra_scaled, 2, current_scaling$sd, "/")

    current_seq_scaled  <- rbind(current_seq_scaled, X_extra_scaled)
    last_seq_idx_global <- i - 1
  }

  # --- 3c) Regime at (i - 1) using only data up to (i - 1) ---
  # The sequence ends at (i - 1), so the smoothed probability at its last index
  # equals the filtered probability: no look-ahead.
  pred_seq <- predict(current_fit, make_hsmm_data(current_seq_scaled), method = "smoothed", M = M_max)
  p_last   <- pred_seq$p[nrow(pred_seq$p), ]

  results[[i - first_pred_idx + 1]] <- data.frame(
    date           = dates_full[i],
    state_ml       = which.max(p_last),
    p_state1       = p_last[1],
    p_state2       = p_last[2],
    trend_state_id = current_trend_state,
    chop_state_id  = current_chop_state,
    trend_prob     = p_last[current_trend_state],
    chop_prob      = p_last[current_chop_state]
  )
}

regimes <- do.call(rbind, results)
write.csv(regimes, "results/hsmm_regimes.csv", row.names = FALSE)
cat("Saved results/hsmm_regimes.csv:", nrow(regimes), "rows\n")
