#! /usr/bin/python
import os
import yaml
import argparse
import subprocess


def parameter_tuning_exp(args, test_script_path):
    weights = [(0.1, 0.3, 0.3, 0.3), (0.2, 0.2, 0.4, 0.2)]
    relax_types = ['sub-binary', 'continuous']
    bodies = ['crayola']
    cs = [0.5, 0.8, 1.0]
    # read template
    with open(args.yaml_template) as template_file:
        yaml_template = template_file.read()
    yaml_folder = os.path.abspath(os.path.dirname(args.yaml_template))
    yaml_instance_filename = yaml_folder + '/yaml_instance.yaml'
    command_list = ["python", test_script_path, yaml_instance_filename, '--num_runs', str(args.num_runs)]
    # run experiments for binary
    # for body in bodies:
    #     for c in cs:
    #         yaml_instance = yaml_template.replace('<BODY_NAME>', body)
    #         yaml_instance = yaml_instance.replace('<RELAX_TYPE>', 'binary')
    #         yaml_instance = yaml_instance.replace('<ARM_WEIGHT>', '0.1')
    #         yaml_instance = yaml_instance.replace('<OBJ_COL_WEIGHT>', '0.3')
    #         yaml_instance = yaml_instance.replace('<CONTACT_WEIGHT>', '0.3')
    #         yaml_instance = yaml_instance.replace('<OBJ_WEIGHT>', '0.3')
    #         yaml_instance = yaml_instance.replace('<C>', str(c))
    #         exp_id = 'binary_' + str(c)
    #         yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
    #         # dump yaml into same folder as template (because of grasps etc)
    #         with open(yaml_instance_filename, 'w') as tmp_yaml_file:
    #             tmp_yaml_file.write(yaml_instance)
    #         # print "Fake call %s" % str(command_list)
    #         subprocess.call(command_list)
    # run experiments for other relaxation types
    for body in bodies:
        for relax in relax_types:
            for ws in weights:
                for c in cs:
                    yaml_instance = yaml_template.replace('<BODY_NAME>', body)
                    yaml_instance = yaml_instance.replace('<RELAX_TYPE>', relax)
                    yaml_instance = yaml_instance.replace('<ARM_WEIGHT>', str(ws[0]))
                    yaml_instance = yaml_instance.replace('<OBJ_COL_WEIGHT>', str(ws[1]))
                    yaml_instance = yaml_instance.replace('<CONTACT_WEIGHT>', str(ws[2]))
                    yaml_instance = yaml_instance.replace('<OBJ_WEIGHT>', str(ws[3]))
                    yaml_instance = yaml_instance.replace('<C>', str(c))
                    exp_id = relax + '_' + str(ws) + '_' + str(c)
                    yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
                    # dump yaml into same folder as template (because of grasps etc)
                    with open(yaml_instance_filename, 'w') as tmp_yaml_file:
                        tmp_yaml_file.write(yaml_instance)
                    # print "Fake call %s" % str(command_list)
                    subprocess.call(command_list)


IROS_BODIES = ['crayola', 'crayola_24', 'wine_glass', 'scaled_table']
IROS_ENVS = ['cabinet_low_clutter', 'cabinet_high_clutter', 'table_high_clutter']
# IROS_BODIES = ['crayola',  'wine_glass']
# IROS_ENVS = ['cabinet_high_clutter', 'table_low_clutter']
# IROS_ENVS = ['cabinet_high_clutter']
PLCMNT_VOLUMES = {
    'cabinet_low_clutter': '[-0.4, 0.45, 0.25, 0.53, 0.87, 0.54]',
    'cabinet_high_clutter': '[-0.4, 0.45, 0.25, 0.53, 0.87, 0.54]',
    'table_low_clutter': '[-0.4, 0.29, 0.0, 0.40, 0.78, 0.2]',
    'table_high_clutter': '[-0.4, 0.29, 0.0, 0.40, 0.78, 0.2]',
}
C_VALS = {
    'sub-binary': str(0.8),
    'binary': str(0.5),
}


def iros_exp_baselines(yaml_template):
    # sampler_types = ['random', 'random_no_opt']
    sampler_types = ['random_no_opt']
    objectives = ['True', 'False']

    def yaml_gen():
        # run experiments for other relaxation types
        for objective in objectives:
            for env in IROS_ENVS:
                for body in IROS_BODIES:
                    for stype in sampler_types:
                        yaml_instance = yaml_template.replace('<ENV_NAME>', env)
                        yaml_instance = yaml_instance.replace('<MAX_CLEARANCE>', objective)
                        yaml_instance = yaml_instance.replace('<BODY_NAME>', body)
                        yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', PLCMNT_VOLUMES[env])
                        yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', stype)
                        exp_id = stype
                        yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
                        yield yaml_instance
    num_batches = len(sampler_types) * len(IROS_ENVS) * len(IROS_BODIES) * len(objectives)
    return yaml_gen(), num_batches


def iros_mcts_simple(yaml_template):
    sampler_types = ['simple_mcts_sampler']
    objectives = ['False', 'True']
    relax_types = ['sub-binary']
    # cs = [0.8]
    cs = [0.5]

    def yaml_gen():
        for objective in objectives:
            for env in IROS_ENVS:
                for body in IROS_BODIES:
                    for stype in sampler_types:
                        for rtype in relax_types:
                            for c in cs:
                                yaml_instance = yaml_template.replace('<ENV_NAME>', env)
                                yaml_instance = yaml_instance.replace('<MAX_CLEARANCE>', objective)
                                yaml_instance = yaml_instance.replace('<BODY_NAME>', body)
                                yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', PLCMNT_VOLUMES[env])
                                yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', stype)
                                yaml_instance = yaml_instance.replace('<RELAX_TYPE>', rtype)
                                yaml_instance = yaml_instance.replace('<C>', str(c))
                                exp_id = stype + '_' + rtype + '_' + '_' + str(c)
                                yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
                                yield yaml_instance
    num_batches = len(sampler_types) * len(IROS_ENVS) * len(IROS_BODIES) * \
        len(objectives) * len(relax_types) * len(cs)
    return yaml_gen(), num_batches


def iros_even_simpler_mcts(yaml_template):
    sampler_types = ['simple_mcts_sampler']
    # objectives = ['False', 'True']
    objectives = ['True']
    relax_types = ['sub-binary']
    # cs = [0.8]
    cs = [0.5]

    def yaml_gen():
        for objective in objectives:
            for env in IROS_ENVS:
                for body in IROS_BODIES:
                    for stype in sampler_types:
                        for rtype in relax_types:
                            for c in cs:
                                yaml_instance = yaml_template.replace('<ENV_NAME>', env)
                                yaml_instance = yaml_instance.replace('<MAX_CLEARANCE>', objective)
                                yaml_instance = yaml_instance.replace('<BODY_NAME>', body)
                                yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', PLCMNT_VOLUMES[env])
                                yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', stype)
                                yaml_instance = yaml_instance.replace('<RELAX_TYPE>', rtype)
                                yaml_instance = yaml_instance.replace('<C>', str(c))
                                exp_id = 'paper_mcts' + '_' + rtype + '_' + '_' + str(c)
                                yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
                                yield yaml_instance
    num_batches = len(sampler_types) * len(IROS_ENVS) * len(IROS_BODIES) * \
        len(objectives) * len(relax_types) * len(cs)
    return yaml_gen(), num_batches


def humanoids_baselines(yaml_template):
    # sampler_types = ['simple_mcts_sampler']
    # objectives = ['False', 'True']
    # objectives = ['False']
    # relax_types = ['sub-binary']
    bodies = ['elmers_glue', 'expo']
    envs = ['table_high_clutter', 'cabinet_high_clutter']
    selector_types = ['naive', 'cache_hierarchy', 'dummy']
    grasp_ids = {'elmers_glue': [0, 1], 'expo': [0]}
    cases_to_skip = [('table_high_clutter', 'elmers_glue', 'naive', 0),
                     ('table_high_clutter', 'elmers_glue', 'dummy', 1),
                     ('cabinet_high_clutter', 'elmers_glue', 'dummy', 1)]

    def yaml_gen():
        for env in envs:
            for body in bodies:
                for seltype in selector_types:
                    for gid in grasp_ids[body]:
                        if (env, body, seltype, gid) in cases_to_skip:
                            print "Skipping :", env, body, seltype, gid
                            continue
                        yaml_instance = yaml_template.replace('<ENV_NAME>', env)
                        yaml_instance = yaml_instance.replace('<OBJECTIVE_FN>', "maximize_clearance")
                        yaml_instance = yaml_instance.replace('<BODY_NAME>', body)
                        yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', PLCMNT_VOLUMES[env])
                        yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', 'simple_mcts_sampler')
                        yaml_instance = yaml_instance.replace('<SELECTOR_TYPE>', seltype)
                        yaml_instance = yaml_instance.replace('<RELAX_TYPE>', 'sub-binary')
                        yaml_instance = yaml_instance.replace('<GRASP_ID>', str(gid))
                        yaml_instance = yaml_instance.replace('<TIME_LIMIT>', '60.0')
                        yaml_instance = yaml_instance.replace('<C>', '0.5')
                        exp_id = 'mg_mcts' + '_' + seltype
                        yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
                        yield yaml_instance
        for seltype in selector_types:
            yaml_instance = yaml_template.replace('<ENV_NAME>', 'cabinet_high_clutter')
            yaml_instance = yaml_instance.replace('<OBJECTIVE_FN>', "deep_shelf")
            yaml_instance = yaml_instance.replace('<BODY_NAME>', 'expo')
            yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', '[-0.4, 0.45, 0.25, -0.14, 1.0, 0.54]')
            yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', 'simple_mcts_sampler')
            yaml_instance = yaml_instance.replace('<SELECTOR_TYPE>', seltype)
            yaml_instance = yaml_instance.replace('<RELAX_TYPE>', 'sub-binary')
            yaml_instance = yaml_instance.replace('<GRASP_ID>', '0')
            yaml_instance = yaml_instance.replace('<TIME_LIMIT>', '180.0')
            yaml_instance = yaml_instance.replace('<C>', '0.5')
            exp_id = 'mg_mcts' + '_' + seltype
            yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
            yield yaml_instance

    num_batches = sum([len(envs) * len(selector_types) * len(grasps) for grasps in grasp_ids.values()]) \
        + len(selector_types)
    return yaml_gen(), num_batches

# def humanoids_addon(yaml_template):
#     bodies = ['expo']
#     envs = ['cabinet_high_clutter']
#     selector_types = ['naive', 'cache_hierarchy', 'dummy']
#     grasp_ids = {'expo': [0]}
#
#     def yaml_gen():
#         for env in envs:
#             for body in bodies:
#                 for seltype in selector_types:
#                     for gid in grasp_ids[body]:
#                         yaml_instance = yaml_template.replace('<ENV_NAME>', env)
#                         yaml_instance = yaml_instance.replace('<OBJECTIVE_FN>', "maximize_clearance")
#                         yaml_instance = yaml_instance.replace('<BODY_NAME>', body)
#                         yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', PLCMNT_VOLUMES[env])
#                         yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', 'simple_mcts_sampler')
#                         yaml_instance = yaml_instance.replace('<SELECTOR_TYPE>', seltype)
#                         yaml_instance = yaml_instance.replace('<RELAX_TYPE>', 'sub-binary')
#                         yaml_instance = yaml_instance.replace('<GRASP_ID>', str(gid))
#                         yaml_instance = yaml_instance.replace('<TIME_LIMIT>', '60.0')
#                         yaml_instance = yaml_instance.replace('<C>', '0.5')
#                         exp_id = 'mg_mcts' + '_' + seltype
#                         yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
#                         yield yaml_instance
#         for seltype in selector_types:
#             yaml_instance = yaml_template.replace('<ENV_NAME>', 'cabinet_high_clutter')
#             yaml_instance = yaml_instance.replace('<OBJECTIVE_FN>', "deep_shelf")
#             yaml_instance = yaml_instance.replace('<BODY_NAME>', 'expo')
#             yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', '[-0.4, 0.45, 0.25, -0.14, 1.0, 0.54]')
#             yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', 'simple_mcts_sampler')
#             yaml_instance = yaml_instance.replace('<SELECTOR_TYPE>', seltype)
#             yaml_instance = yaml_instance.replace('<RELAX_TYPE>', 'sub-binary')
#             yaml_instance = yaml_instance.replace('<GRASP_ID>', '0')
#             yaml_instance = yaml_instance.replace('<TIME_LIMIT>', '180.0')
#             yaml_instance = yaml_instance.replace('<C>', '0.5')
#             exp_id = 'mg_mcts' + '_' + seltype
#             yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
#             yield yaml_instance
#
#     num_batches = sum([len(envs) * len(selector_types) * len(grasps) for grasps in grasp_ids.values()]) \
#         + len(selector_types)
#     return yaml_gen(), num_batches
def humanoids_addon(yaml_template):
    bodies = ['expo']
    envs = ['cabinet_high_clutter']
    selector_types = ['naive', 'cache_hierarchy', 'dummy']
    grasp_ids = {'expo': [0]}

    def yaml_gen():
        for seltype in selector_types:
            yaml_instance = yaml_template.replace('<ENV_NAME>', 'cabinet_high_clutter')
            yaml_instance = yaml_instance.replace('<OBJECTIVE_FN>', "deep_shelf_min")
            yaml_instance = yaml_instance.replace('<BODY_NAME>', 'expo')
            yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', '[-0.4, 0.45, 0.25, -0.14, 1.0, 0.54]')
            yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', 'simple_mcts_sampler')
            yaml_instance = yaml_instance.replace('<SELECTOR_TYPE>', seltype)
            yaml_instance = yaml_instance.replace('<RELAX_TYPE>', 'sub-binary')
            yaml_instance = yaml_instance.replace('<GRASP_ID>', '0')
            yaml_instance = yaml_instance.replace('<TIME_LIMIT>', '180.0')
            yaml_instance = yaml_instance.replace('<C>', '0.5')
            exp_id = 'mg_mcts' + '_' + seltype
            yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
            yield yaml_instance

    num_batches = sum([len(envs) * len(selector_types) * len(grasps) for grasps in grasp_ids.values()]) \
        + len(selector_types)
    return yaml_gen(), num_batches

# def iros_mcts(yaml_template):
#     sampler_types = ['mcts_sampler']
#     objectives = ['False', 'True']
#     relax_types = ['binary']
#     # projection_type = ['None', 'ik', 'jac']
#     projection_type = ['None']
#     cs = [0.5, 0.8]

#     def yaml_gen():
#         for objective in objectives:
#             for env in IROS_ENVS:
#                 for body in IROS_BODIES:
#                     for stype in sampler_types:
#                         for rtype in relax_types:
#                             for ptype in projection_type:
#                                 for c in cs:
#                                     yaml_instance = yaml_template.replace('<ENV_NAME>', env)
#                                     yaml_instance = yaml_instance.replace('<MAX_CLEARANCE>', objective)
#                                     yaml_instance = yaml_instance.replace('<BODY_NAME>', body)
#                                     yaml_instance = yaml_instance.replace('<PLCMNT_VOL>', PLCMNT_VOLUMES[env])
#                                     yaml_instance = yaml_instance.replace('<SAMPLER_TYPE>', stype)
#                                     yaml_instance = yaml_instance.replace('<RELAX_TYPE>', rtype)
#                                     yaml_instance = yaml_instance.replace('<C>', str(c))
#                                     yaml_instance = yaml_instance.replace('<PROJ_TYPE>', ptype)
#                                     exp_id = stype + '_' + rtype + '_' + ptype
#                                     yaml_instance = yaml_instance.replace('<EXP_ID>', exp_id)
#                                     yield yaml_instance
#     num_batches = len(sampler_types) * len(IROS_ENVS) * len(IROS_BODIES) * \
#         len(objectives) * len(relax_types) * len(projection_type) * len(cs)
#     return yaml_gen(), num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run placement experiments specified by a list of yaml files")
    parser.add_argument('yaml_template', type=str, help='Path to a file containing problem definition template')
    # parser.add_argument('options', type=str, help='Path to a yaml file containing parameter options')
    parser.add_argument('num_runs', type=int, help='Number of runs for each problem')
    parser.add_argument('id', type=int, help='Experiments id')
    parser.add_argument('offset', type=int, help='Id offset')
    parser.add_argument('command', type=str, default='/test_scripts/test_placement2.py',
                        help='Path to script to execute relative to this script\'s path')
    args = parser.parse_args()
    test_script_path = os.path.abspath(os.path.dirname(__file__)) + args.command
    # read template
    with open(args.yaml_template) as template_file:
        yaml_template = template_file.read()
    yaml_folder = os.path.abspath(os.path.dirname(args.yaml_template))
    yaml_instance_filename = yaml_folder + '/yaml_instance' + str(args.id) + '.yaml'
    command_list = ["python", test_script_path, yaml_instance_filename,
                    '--num_runs', str(args.num_runs), '--offset', str(args.offset)]
    # yaml_gen, num_batches = iros_exp_baselines(yaml_template)
    # yaml_gen, num_batches = iros_mcts_simple(yaml_template)
    # yaml_gen, num_batches = iros_mcts_simple(yaml_template)
    # yaml_gen, num_batches = iros_mcts_simple(yaml_template)
    # yaml_gen, num_batches = iros_even_simpler_mcts(yaml_template)
    # yaml_gen, num_batches = humanoids_baselines(yaml_template)
    yaml_gen, num_batches = humanoids_addon(yaml_template)
    for batch_id in xrange(num_batches):
        try:
            yaml_instance = yaml_gen.next()
            with open(yaml_instance_filename, 'w') as tmp_yaml_file:
                tmp_yaml_file.write(yaml_instance)
            print "Running batch %i/%i" % (batch_id + 1, num_batches)
            subprocess.call(command_list)
        except StopIteration:
            break
    print "Experiments finished"
