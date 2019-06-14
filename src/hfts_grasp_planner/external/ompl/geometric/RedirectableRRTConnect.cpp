/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2008, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Author: Ioan Sucan */

#include "ompl/geometric/planners/rrt/RedirectableRRTConnect.h"
// #include <iostream>
#include "ompl/base/goals/GoalSampleableRegion.h"
#include "ompl/tools/config/SelfConfig.h"

ompl::geometric::RedirectableRRTConnect::RedirectableRRTConnect(const base::SpaceInformationPtr &si,
                                                                bool addIntermediateStates)
  : base::Planner(si, addIntermediateStates ? "RedirectableRRTConnectIntermediate" : "RedirectableRRTConnect")
{
    specs_.recognizedGoal = base::GOAL_SAMPLEABLE_REGION;
    specs_.directed = true;

    Planner::declareParam<double>("range", this, &RedirectableRRTConnect::setRange, &RedirectableRRTConnect::getRange,
                                  "0.:1.:10000.");
    Planner::declareParam<bool>("intermediate_states", this, &RedirectableRRTConnect::setIntermediateStates,
                                &RedirectableRRTConnect::getIntermediateStates, "0,1");

    distanceBetweenTrees_ = std::numeric_limits<double>::infinity();
    addIntermediateStates_ = addIntermediateStates;
}

ompl::geometric::RedirectableRRTConnect::~RedirectableRRTConnect()
{
    freeMemory();
}

void ompl::geometric::RedirectableRRTConnect::setup()
{
    Planner::setup();
    tools::SelfConfig sc(si_, getName());
    sc.configurePlannerRange(maxDistance_);

    if (!tStart_)
        tStart_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    if (!tGoal_)
        tGoal_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));  // reset of goal tree!
    if (!goalTrees_)
        goalTrees_.reset(tools::SelfConfig::getDefaultNearestNeighbors<std::shared_ptr<GoalTreeSubset>>(this));
    tStart_->setDistanceFunction([this](const Motion *a, const Motion *b) { return distanceFunction(a, b); });
    tGoal_->setDistanceFunction([this](const Motion *a, const Motion *b) { return distanceFunction(a, b); });
    goalTrees_->setDistanceFunction(
        [this](const std::shared_ptr<const GoalTreeSubset> a, const std::shared_ptr<const GoalTreeSubset> b) {
            return treeSetDistanceFunction(a, b);
        });
    queryTreeSet_ = std::make_shared<GoalTreeSubset>(new Motion(si_));
    num_goal_tree_samples_ = 0;
}

void ompl::geometric::RedirectableRRTConnect::freeMemory()
{
    std::vector<Motion *> motions;

    if (tStart_)
    {
        tStart_->list(motions);
        for (auto &motion : motions)
        {
            if (motion->state != nullptr)
                si_->freeState(motion->state);
            delete motion;
        }
    }

    if (tGoal_)
    {
        tGoal_->list(motions);
        for (auto &motion : motions)
        {
            if (motion->state != nullptr)
                si_->freeState(motion->state);
            delete motion;
        }
    }

    if (goalTrees_)
    {
        std::vector<std::shared_ptr<GoalTreeSubset>> trees;
        goalTrees_->list(trees);
        for (auto &tree : trees)
        {
            tree->nodes.clear();  // elements of this have already been deleted above
        }
    }

    if (queryTreeSet_)
    {
        if (queryTreeSet_->root->state != nullptr)
            si_->freeState(queryTreeSet_->root->state);
        delete queryTreeSet_->root;
        queryTreeSet_->nodes.clear();
    }
}

void ompl::geometric::RedirectableRRTConnect::clear()
{
    Planner::clear();
    sampler_.reset();
    freeMemory();
    if (tStart_)
        tStart_->clear();
    if (tGoal_)
        tGoal_->clear();
    if (goalTrees_)
        goalTrees_->clear();
    distanceBetweenTrees_ = std::numeric_limits<double>::infinity();
    num_goal_tree_samples_ = 0;
}

void ompl::geometric::RedirectableRRTConnect::addToBackwardTreeSet(Motion *m)
{
    auto goal_tree = getTreeSet(m->root);
    assert(goal_tree != nullptr);
    goal_tree->nodes.push_back(m);
    if (goal_tree->is_goal)
    {
        num_goal_tree_samples_ += 1;
    }
}

std::pair<bool, ompl::geometric::RedirectableRRTConnect::Motion *>
ompl::geometric::RedirectableRRTConnect::mergeIntoForwardTree(ompl::geometric::RedirectableRRTConnect::Motion *fm,
                                                              ompl::geometric::RedirectableRRTConnect::Motion *m)
{
    // std::cout << "Merging backward tree into forward tree." << std::endl;
    assert(si_->distance(fm->state, m->state) == 0.0);
    // first retrieve backwards tree set of backward tree
    auto btree_set = getTreeSet(m->root);
    assert(btree_set != nullptr);

    // rewire path from m's parent to root
    Motion *bcurrent = m->parent;
    Motion *fcurrent = fm;
    while (bcurrent != nullptr)
    {
        Motion *bparent = bcurrent->parent;  // save parent of bcurrent
        // rewire bcurrent to forward tree
        bcurrent->parent = fcurrent;
        bcurrent->root = fcurrent->root;
        // move on to next
        fcurrent = bcurrent;
        bcurrent = bparent;
    }
    // move all motions of the backward tree to the forward tree
    bool is_goal = btree_set->is_goal;
    if (is_goal)
    {
        // std::cout << "The backward tree is a goal. Having " << num_goal_tree_samples_ << " samples. Removing "
        //   << btree_set->nodes.size() << std::endl;
        num_goal_tree_samples_ -= btree_set->nodes.size();
        // std::cout << "Number of goal samples left:" << num_goal_tree_samples_ << std::endl;
    }
    for (auto *motion : btree_set->nodes)
    {
        tGoal_->remove(motion);
        if (motion != m)
            tStart_->add(motion);
    }
    // delete tree set
    goalTrees_->remove(btree_set);
    // delete m
    si_->freeState(m->state);
    delete m;
    return std::make_pair(is_goal, fcurrent);
}

ompl::geometric::RedirectableRRTConnect::GrowState
ompl::geometric::RedirectableRRTConnect::growTree(TreeData &tree, TreeGrowingInfo &tgi, Motion *rmotion)
{
    /* find closest state in the tree */
    Motion *nmotion = tree->nearest(rmotion);

    /* assume we can reach the state we go towards */
    bool reach = true;

    /* find state to add */
    base::State *dstate = rmotion->state;
    double d = si_->distance(nmotion->state, rmotion->state);
    if (d > maxDistance_)
    {
        si_->getStateSpace()->interpolate(nmotion->state, rmotion->state, maxDistance_ / d, tgi.xstate);

        /* Check if we have moved at all. Due to some stranger state spaces (e.g., the constrained state spaces),
         * interpolate can fail and no progress is made. Without this check, the algorithm gets stuck in a loop as it
         * thinks it is making progress, when none is actually occurring. */
        if (si_->equalStates(nmotion->state, tgi.xstate))
            return TRAPPED;

        dstate = tgi.xstate;
        reach = false;
    }

    bool validMotion = tgi.start ? si_->checkMotion(nmotion->state, dstate) :
                                   si_->isValid(dstate) && si_->checkMotion(dstate, nmotion->state);

    if (!validMotion)
        return TRAPPED;

    if (addIntermediateStates_)
    {
        const base::State *astate = tgi.start ? nmotion->state : dstate;
        const base::State *bstate = tgi.start ? dstate : nmotion->state;

        std::vector<base::State *> states;
        const unsigned int count = si_->getStateSpace()->validSegmentCount(astate, bstate);

        if (si_->getMotionStates(astate, bstate, states, count, true, true))
            si_->freeState(states[0]);

        for (std::size_t i = 1; i < states.size(); ++i)
        {
            Motion *motion = new Motion;
            motion->state = states[i];
            motion->parent = nmotion;
            motion->root = nmotion->root;
            tree->add(motion);
            if (!tgi.start)
            {
                addToBackwardTreeSet(motion);
            }

            nmotion = motion;
        }

        tgi.xmotion = nmotion;
    }
    else
    {
        Motion *motion = new Motion(si_);
        si_->copyState(motion->state, dstate);
        motion->parent = nmotion;
        motion->root = nmotion->root;
        tree->add(motion);
        if (!tgi.start)
        {
            addToBackwardTreeSet(motion);
        }

        tgi.xmotion = motion;
    }

    return reach ? REACHED : ADVANCED;
}

ompl::base::PlannerStatus ompl::geometric::RedirectableRRTConnect::solve(const base::PlannerTerminationCondition &ptc)
{
    checkValidity();
    auto *goal = dynamic_cast<base::GoalSampleableRegion *>(pdef_->getGoal().get());

    if (goal == nullptr)
    {
        OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
        return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
    }

    while (const base::State *st = pis_.nextStart())
    {
        auto *motion = new Motion(si_);
        si_->copyState(motion->state, st);
        motion->root = motion->state;
        tStart_->add(motion);
    }

    if (tStart_->size() == 0)
    {
        OMPL_ERROR("%s: Motion planning start tree could not be initialized!", getName().c_str());
        return base::PlannerStatus::INVALID_START;
    }

    if (num_goal_tree_samples_ == 0 and !goal->couldSample())
    {
        OMPL_ERROR("%s: Insufficient states in sampleable goal region", getName().c_str());
        return base::PlannerStatus::INVALID_GOAL;
    }

    if (!sampler_)
        sampler_ = si_->allocStateSampler();

    OMPL_INFORM("%s: Starting planning with %d states already in datastructure", getName().c_str(),
                (int)(tStart_->size() + tGoal_->size()));

    TreeGrowingInfo tgi;
    tgi.xstate = si_->allocState();

    // Motion *approxsol = nullptr;
    // double approxdif = std::numeric_limits<double>::infinity();
    auto *rmotion = new Motion(si_);
    base::State *rstate = rmotion->state;
    bool startTree = true;
    bool solved = false;
    unsigned int num_goals_sampled = 0;

    while (!ptc)
    {
        TreeData &tree = startTree ? tStart_ : tGoal_;
        tgi.start = startTree;
        startTree = !startTree;
        TreeData &otherTree = startTree ? tStart_ : tGoal_;

        // TODO this is specific to placement planner use case
        if (tGoal_->size() == 0 ||
            num_goals_sampled < goal->maxSampleCount())  // not using pis_ here because it would need to be reset
        {
            // we have a new goal
            auto *motion = new Motion(si_);
            goal->sampleGoal(motion->state);
            motion->root = motion->state;
            tGoal_->add(motion);
            auto new_tree = std::make_shared<GoalTreeSubset>(motion);
            goalTrees_->add(new_tree);
            num_goal_tree_samples_ += 1;
            num_goals_sampled += 1;
        }

        /* sample random state */
        sampler_->sampleUniform(rstate);

        GrowState gs = growTree(tree, tgi, rmotion);

        if (gs != TRAPPED)
        {
            /* remember which motion was just added */
            Motion *addedMotion = tgi.xmotion;

            /* attempt to connect trees */

            /* if reached, it means we used rstate directly, no need top copy again */
            if (gs != REACHED)
                si_->copyState(rstate, tgi.xstate);

            GrowState gsc = ADVANCED;
            tgi.start = startTree;
            while (gsc == ADVANCED)
                gsc = growTree(otherTree, tgi, rmotion);

            /* update distance between trees */
            const double newDist = tree->getDistanceFunction()(addedMotion, otherTree->nearest(addedMotion));
            if (newDist < distanceBetweenTrees_)
            {
                distanceBetweenTrees_ = newDist;
                // OMPL_INFORM("Estimated distance to go: %f", distanceBetweenTrees_);
            }

            Motion *startMotion = startTree ? tgi.xmotion : addedMotion;
            Motion *goalMotion = startTree ? addedMotion : tgi.xmotion;

            /* if we connected the trees in a valid way (start and goal pair is valid)*/
            if (gsc == REACHED && goal->isStartGoalPairValid(startMotion->root, goalMotion->root))
            {
                // in any case merge the connected backward and forward tree
                bool b_goal;
                Motion *solution;
                std::tie(b_goal, solution) = mergeIntoForwardTree(startMotion, goalMotion);
                if (b_goal)
                {
                    // std::cout << "The tree was a goal tree!" << std::endl;
                    /* construct the solution path */
                    std::vector<Motion *> mpath;
                    while (solution != nullptr)
                    {
                        mpath.push_back(solution);
                        solution = solution->parent;
                    }

                    auto path(std::make_shared<PathGeometric>(si_));
                    path->getStates().reserve(mpath.size());
                    for (int i = mpath.size() - 1; i >= 0; --i)
                        path->append(mpath[i]->state);

                    pdef_->addSolutionPath(path, false, 0.0, getName());
                    solved = true;
                    break;
                }
            }
            // else
            // {
            //     // We didn't reach the goal, but if we were extending the start
            //     // tree, then we can mark/improve the approximate path so far.
            //     if (!startTree)
            //     {
            //         // We were working from the startTree.
            //         double dist = 0.0;
            //         goal->isSatisfied(tgi.xmotion->state, &dist);
            //         if (dist < approxdif)
            //         {
            //             approxdif = dist;
            //             approxsol = tgi.xmotion;
            //         }
            //     }
            // }
        }
    }

    si_->freeState(tgi.xstate);
    si_->freeState(rstate);
    delete rmotion;

    OMPL_INFORM("%s: Created %u states (%u start + %u goal)", getName().c_str(), tStart_->size() + tGoal_->size(),
                tStart_->size(), tGoal_->size());

    // if (approxsol && !solved)
    // {
    //     /* construct the solution path */
    //     std::vector<Motion *> mpath;
    //     while (approxsol != nullptr)
    //     {
    //         mpath.push_back(approxsol);
    //         approxsol = approxsol->parent;
    //     }

    //     auto path(std::make_shared<PathGeometric>(si_));
    //     for (int i = mpath.size() - 1; i >= 0; --i)
    //         path->append(mpath[i]->state);
    //     pdef_->addSolutionPath(path, true, approxdif, getName());
    //     return base::PlannerStatus::APPROXIMATE_SOLUTION;
    // }

    return solved ? base::PlannerStatus::EXACT_SOLUTION : base::PlannerStatus::TIMEOUT;
}

void ompl::geometric::RedirectableRRTConnect::removeGoals(const std::vector<const base::State *> &old_goals)
{
    for (auto &old_goal : old_goals)  // for each goal
    {
        // check whether we have a tree set for this goal
        auto goal_tree = getTreeSet(old_goal);
        // if (goal_tree != nullptr and goal_tree->is_goal)
        if (goal_tree != nullptr)
        {
            if (!goal_tree->is_goal)
                throw std::logic_error("Removing a goal tree twice!");
            goal_tree->is_goal = false;  // mark it as non-goal
            num_goal_tree_samples_ -= goal_tree->nodes.size();
        }
    }
}

void ompl::geometric::RedirectableRRTConnect::removeGoals(const std::vector<base::ScopedState<>> &old_goals)
{
    std::vector<const base::State *> raw_pointers;
    raw_pointers.reserve(old_goals.size());
    for (auto &scoped_state : old_goals)
    {
        raw_pointers.push_back(scoped_state.get());
    }
    removeGoals(raw_pointers);
}

void ompl::geometric::RedirectableRRTConnect::getGoals(std::vector<base::ScopedState<>> &current_goals)
{
    std::vector<std::shared_ptr<GoalTreeSubset>> goal_trees;
    goalTrees_->list(goal_trees);
    for (auto gtree : goal_trees)
    {
        if (gtree->is_goal)
            current_goals.emplace_back(base::ScopedState<>(si_->getStateSpace(), gtree->root->state));
    }
}

void ompl::geometric::RedirectableRRTConnect::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion *> motions;
    if (tStart_)
        tStart_->list(motions);

    for (auto &motion : motions)
    {
        if (motion->parent == nullptr)
            data.addStartVertex(base::PlannerDataVertex(motion->state, 1));
        else
        {
            data.addEdge(base::PlannerDataVertex(motion->parent->state, 1), base::PlannerDataVertex(motion->state, 1));
        }
    }

    motions.clear();
    if (tGoal_)
        tGoal_->list(motions);

    for (auto &motion : motions)
    {
        if (motion->parent == nullptr)
            data.addGoalVertex(base::PlannerDataVertex(motion->state, 2));
        else
        {
            // The edges in the goal tree are reversed to be consistent with start tree
            data.addEdge(base::PlannerDataVertex(motion->state, 2), base::PlannerDataVertex(motion->parent->state, 2));
        }
    }

    // Add some info.
    data.properties["approx goal distance REAL"] = boost::lexical_cast<std::string>(distanceBetweenTrees_);
}

std::shared_ptr<ompl::geometric::RedirectableRRTConnect::GoalTreeSubset>
ompl::geometric::RedirectableRRTConnect::getTreeSet(const base::State *root) const
{
    assert(queryTreeSet_ != nullptr);
    // check whether we have a tree set
    si_->copyState(queryTreeSet_->root->state, root);
    auto nearest_goal_tree = goalTrees_->nearest(queryTreeSet_);
    if (treeSetDistanceFunction(queryTreeSet_, nearest_goal_tree) == 0.0)
    {
        return nearest_goal_tree;
    }
    return nullptr;
}

void ompl::geometric::RedirectableRRTConnect::removeTreeSet(const base::State *root)
{
    auto tree_set = getTreeSet(root);
    if (tree_set != nullptr)
    {
        goalTrees_->remove(tree_set);
    }
}
