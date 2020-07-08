import mlflow
import numpy as np


def log_values(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)

def persist_train_progress(cost, epoch, log_likelihood, reinforce_loss):
    avg_cost = cost.mean().item()
    mlflow.log_metric("avg_cost", float(avg_cost), step=epoch)
    mlflow.log_metric("actor_loss", float(reinforce_loss.item()), step=epoch)
    mlflow.log_metric("nll", float(-log_likelihood.mean().item()), step=epoch)


def persist_run_params(opts):
    mlflow.log_param("graph_size", opts.graph_size)
    mlflow.log_param("batch_size", opts.batch_size)
    mlflow.log_param("epoch_size", opts.epoch_size)
    mlflow.log_param("val_size", opts.val_size)
    #  mlflow.log_param("shift_time", opts.shift_time)
    #  mlflow.log_param("smallest_task_time", opts.smallest_task_time)
    #  mlflow.log_param("is_dynamic_embed", opts.is_dynamic_embed)

def persist_final_training_score(costs, greedy_costs):
    mean_model_cost, std_model_cost = np.mean(costs), 2 * np.std(costs) / np.sqrt(len(costs))
    mean_greedy_cost, std_greedy_cost = np.mean(greedy_costs), 2 * np.std(greedy_costs) / np.sqrt(len(greedy_costs))
    mlflow.log_param("model_mean_cost", mean_model_cost)
    mlflow.log_param("greedy_mean_cost", mean_greedy_cost)
    mlflow.log_param("std_model_cost", std_model_cost)
    mlflow.log_param("std_greedy_cost", std_greedy_cost)
