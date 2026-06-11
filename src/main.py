import click
from runs import RunFactory


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--run', '-r', default='compas_fairlab', help='Run to execute')
@click.option('--project_name', '-p', default='CompasFairLab', help='Project name')
@click.option('--root_dir', default=None, help='Root directory containing dataset folders')
@click.option('--start_index', '-s', default=1, help='Start index')
@click.option('--id', '-i', default='test', help='Run id')
@click.option('-metrics_list', '-ml', multiple=True, help='List of metrics')
@click.option('-groups_list', '-gl', multiple=True, help='List of groups')
@click.option('-threshold_list', '-tl', type=float, multiple=True, help='List of threshold')
@click.option('--num_subproblems', '-ns', default=1, help='Number of subproblems')
@click.option('--num_global_iterations', '-ng', default=30, help='Number of global iterations')
@click.option('--num_local_iterations', '-nl', default=30, help='Number of local iterations')
@click.option('--performance_budget', '-pb',default=1.0, help='Performance constraint')
@click.option('--delta', '-d', default=0.02, help='Delta')
@click.option('--delta_step', '-ds', default=0.01, help='Delta')
@click.option('--delta_tol', '-dt', default=0.05, help='Delta')
@click.option('--max_constraints', '-mc', default=10000, help='Max constraints')
@click.option('--global_patience', '-gp', default=5, help='Global (Orchestrator) patience')
@click.option('--local_patience', '-lp', default=5, help='Local Learner patience')
@click.option('--epsilon', '-e', default=0.0, help='Privacy budget epsilon')
@click.option('--num_classes', default=2, help='Number of classes (for classification tasks)')
@click.option('--explainer_epochs', default=None, type=int, help='Number of epochs used to train the FastSHAP explainer')
def main(run, project_name, root_dir, start_index, id,
         metrics_list, groups_list, threshold_list,
         num_subproblems, num_global_iterations, 
         num_local_iterations,performance_budget,
         delta,delta_step,delta_tol,
         max_constraints,global_patience, local_patience, 
         num_classes,epsilon,explainer_epochs):

    run = RunFactory.create_run(run,
                                project_name=project_name,
                                root_dir=root_dir,
                                start_index=start_index,
                                id=id,
                                metrics_list=metrics_list,
                                groups_list=groups_list,
                                threshold_list=threshold_list,
                                num_subproblems=num_subproblems,
                                num_global_iterations=num_global_iterations,
                                num_local_iterations=num_local_iterations,
                                performance_constraint=performance_budget,
                                delta=delta,
                                delta_step=delta_step,
                                delta_tol=delta_tol,
                                max_constraints_in_subproblem=max_constraints,
                                global_patience=global_patience,
                                local_patience=local_patience,
                                epsilon=epsilon,
                                num_classes=num_classes,
                                explainer_epochs=explainer_epochs
                                )
    run()


if __name__ == '__main__':
    # mp.set_start_method("spawn", force=True)
    main()
