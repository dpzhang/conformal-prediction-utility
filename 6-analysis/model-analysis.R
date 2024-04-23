mcmc_convergence_check = function(brms_fit){
  # Check convergence Rhat and neff
  fit_rhats = brms_fit %>% brms::rhat()
  fit_neffs = brms_fit %>% brms::neff_ratio() 
  
  rhats_dist = fit_rhats %>% 
    mcmc_rhat_hist(binwidth = 0.0001) + 
    labs(title = 'Distribution of Rhat')
  neff_dist = fit_neffs %>% 
    mcmc_neff_hist(binwidth = 0.01) +
    labs(title = 'Distribution of Neff')
  plot_eval = (rhats_dist | neff_dist)
   
  output = list(rhats = fit_rhats, neffs = fit_neffs,
                plot = plot_eval)
  return(output)
}

posterior_predictive_check = function(brms_fit){
  response_var = brms_fit$formula %>% 
    as.character %>% first %>% str_split('~') %>% first %>% first

  response_var = ifelse(response_var == 'acc', "Accuracy", "SP")
  
  # Check posterior predictive fit posterior predictive fit
  a = pp_check(brms_fit, type = "bars", ndraws = 1000) + 
    labs(title = str_glue("{response_var} PP: Pooled"))
  b = pp_check(brms_fit, type = "bars_grouped", group = "condition", ndraws = 1000) +
    labs(title = str_glue("{response_var} PP: Condition"))
  c = pp_check(brms_fit, type = "bars_grouped", group = "ioblock", ndraws = 1000) + 
    labs(title = str_glue("{response_var} PP: I/O"))
  d = pp_check(brms_fit, type = "bars_grouped", group = "dbucket", ndraws = 1000) + 
    labs(title = str_glue("{response_var} PP: Difficulty"))
  e = pp_check(brms_fit, type = "bars_grouped", group = "sbucket", ndraws = 1000) +
    labs(title = str_glue("{response_var} PP: RAPS Size"))
  outputPlot = (a | b) /
    (c | d | e)
  
  return(outputPlot)
}


generate_counterfactual = function(brms_fit){
  # Create counterfactual scenario
  # set trial value to the mean as we focus on the effects of other predictors 
  # while holding trial constant, which is common for analyzing marginal effects.
  scenario = expand.grid(condition = c(0, 1, 2, 3),
                         ioblock = c(0, 1),
                         dbucket = c(0, 1),
                         sbucket = c(0, 1),
                         trial = mean(1:16))
  
  # Generate sample draws for different category
  epreds = brms_fit %>% 
    # for each unique combination of predictor values in scenario, we obtain 1000 expected predictions based on 
    # 1000 draws from the posterior distribution of the model's fixed effects. 
    add_epred_draws(newdata = scenario, re_formula = NA, 
                    ndraws = 1000, seed = 42) 
  
  # Now decorate epreds to be something more readable
  drawsDF = epreds %>% 
    mutate(condition = case_when(condition == 0 ~ "Baseline",
                                 condition == 1 ~ "Top-1",
                                 condition == 2 ~ "Top-10",
                                 condition == 3 ~ "RAPS") %>% 
             factor(levels = c('Baseline', 'Top-1', 'Top-10', 'RAPS')),
           ioblock = case_when(ioblock == 0 ~ "In-distribution",
                               ioblock == 1 ~ "Out-of-distribution") %>% 
             factor(levels = c('In-distribution', 'Out-of-distribution')),
           dbucket = case_when(dbucket == 0 ~ "Easy",
                               dbucket == 1 ~ "Hard") %>% 
             factor(levels = c('Easy', 'Hard')),
           sbucket = case_when(sbucket == 0 ~ "Small",
                               sbucket == 1 ~ "Large") %>% 
             factor(levels = c('Small', 'Large')))
  
  
  return(drawsDF)
}


compute_prediction_accuracy = function(){
  taskInfoDF = read.csv("../1-stimuli/stimuli.csv")
  accPred = taskInfoDF %>% 
    select(io, difficulty, size, top1_acc, top10_acc, raps_coverage) %>% 
    group_by(io, difficulty, size) %>% 
    summarize(`Top-1` = mean(top1_acc),
              `Top-10` = mean(top10_acc), 
              `RAPS` = mean(raps_coverage), 
              .groups = 'drop') %>%
    mutate(io = io %>% factor(levels = c('In-distribution', 'Out-of-distribution')),
           difficulty = difficulty %>% factor(levels = c('Easy', 'Hard')),
           size = size %>% factor(levels = c('Small', 'Large'))) %>% 
    pivot_longer(cols = c(`Top-1`, `Top-10`, `RAPS`), names_to = 'condition', values_to = 'acc') %>%
    mutate(condition = factor(condition, levels = c('Baseline', 'Top-1', 'Top-10', 'RAPS'))) %>%
    rename(ioblock = io, dbucket = difficulty, sbucket = size)
  
  return(accPred)
}

plot_marginal_effects = function(drawsDF, var = c('acc', 'sp'), block = c("in", "out")){
  accPred = compute_prediction_accuracy()
  
  io = ifelse(block == 'in', 'In-distribution', 'Out-of-distribution')
  accPred %<>% filter(ioblock == io)
  
  marginal_plot = drawsDF %>%
    filter(ioblock == io) %>% 
    ggplot(aes(x = condition, y = .epred)) +
    stat_halfeye(aes(slab_color = condition, slab_fill = condition,
                     interval_color = condition, point_color = condition), 
                 slab_alpha = 0.5, stroke = 0.1,
                 point_interval = 'median_hdi') + 
    facet_grid(cols = vars(ioblock, dbucket, sbucket)) +
    theme_clean() + 
    theme(legend.position = "None",
          axis.text=element_text(size=12),
          axis.title=element_text(size=15),
          plot.title = element_text(size = 20),
          strip.text.x = element_text(size = 15, face = "bold.italic"),
          strip.text.y = element_text(size = 15, face = "bold.italic"),
          plot.background = element_blank()) + 
    scale_color_manual(
      values = tableau10,
      aesthetics = c("slab_color"),
    ) +
    scale_color_manual(
      values = tableau10,
      aesthetics = c("interval_color"),
    ) + 
    scale_color_manual(
      values = tableau10,
      aesthetics = c("point_color"),
    ) + 
    scale_fill_manual(
      values = tableau10,
      aesthetics = "slab_fill",
    )
  
  if (var == 'acc'){
    marginal_plot = marginal_plot + 
      geom_errorbar(data = accPred, 
                    aes(y = acc, ymin = acc, ymax = acc, color = condition),
                    position = position_nudge(x = 0.5),
                    linewidth = 1) +
      scale_color_manual(
        values = tableau10[-1],
        aesthetics = c("color"),
      )  
  }
  
  return(marginal_plot)
}

plot_all_effects = function(drawsDF, var = c('acc', 'sp')){
  accPred = compute_prediction_accuracy()
  marginal_plot = drawsDF %>%
    ggplot(aes(x = condition, y = .epred)) +
    stat_halfeye(aes(slab_color = condition, slab_fill = condition,
                     interval_color = condition, point_color = condition), 
                 slab_alpha = 0.5, stroke = 0.1,
                 point_interval = 'median_hdi') + 
    facet_grid(cols = vars(dbucket, sbucket),
               rows = vars(ioblock)) +
    theme_clean() + 
    theme(legend.position = "None",
          axis.text=element_text(size=10),
          axis.title=element_text(size=15),
          plot.title = element_text(size = 20),
          strip.text.x = element_text(size = 15, face = "bold.italic"),
          strip.text.y = element_text(size = 15, face = "bold.italic"),
          plot.background = element_blank()) + 
    scale_color_manual(
      values = tableau10,
      aesthetics = c("slab_color"),
    ) +
    scale_color_manual(
      values = tableau10,
      aesthetics = c("interval_color"),
    ) + 
    scale_color_manual(
      values = tableau10,
      aesthetics = c("point_color"),
    ) + 
    scale_fill_manual(
      values = tableau10,
      aesthetics = "slab_fill",
    )
  
  if (var == 'acc'){
    marginal_plot = marginal_plot + 
      geom_errorbar(data = accPred, 
                    aes(y = acc, ymin = acc, ymax = acc, color = condition),
                    position = position_nudge(x = 0.5),
                    linewidth = 1) +
      scale_color_manual(
        values = tableau10[-1],
        aesthetics = c("color"),
      )  
  }
  
  return(marginal_plot)
}


generate_effect = function(drawsDF, group_var, response_var, interval_func){
  # Convert group_var to a list of symbols
  group_symbols = rlang::syms(group_var)
  
  effectDF = drawsDF %>% 
    ungroup %>% 
    group_by(!!!group_symbols) %>%
    interval_func(!!sym(response_var)) 
  
  return(effectDF)
}

save_plot = function(filename, ggplotObj, fig_width = 12, fig_height = 5){
  ggsave(filename = str_glue('../5-paper/figures/{filename}'),
         plot = ggplotObj, 
         device = 'png', dpi = 'print', 
         width = fig_width, height = fig_height)
}

