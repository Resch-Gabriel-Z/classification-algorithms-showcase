# function to print the tree, and return the values, thresholds, and feature indices of the nodes, so we can plot the tree
def print_tree(node, spacing=" ", depth=0, direction="root", tree_strings=None):
    if tree_strings is None:
        tree_strings = []

    # print tree recursively

    if node.value is not None:
        tree_strings.append(f"🍃, {direction}\n{spacing*(depth+1)}**Depth:** {depth}\n**Predict:** {node.value}")
        tree_strings.append(f'---\n')
    else:
        tree_strings.append(f"🌲, {direction}\n{spacing*(depth+1)}**Depth:** {depth}\n**Feature:** {node.feature_index} <= {round(node.threshold, 3)}")
        tree_strings.append(f'---\n')
        # print the left subtree
        print_tree(
            node.left,
            spacing + " > ",
            depth + 1,
            direction="left",
            tree_strings=tree_strings,
        )

        # print the right subtree
        print_tree(
            node.right,
            spacing + " > ",
            depth + 1,
            direction="right",
            tree_strings=tree_strings,
        )

    return tree_strings
