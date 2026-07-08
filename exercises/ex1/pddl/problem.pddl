(define (problem push-prob-2-small-1-heavy)
  (:domain box-push)
  (:objects
    agent_0 agent_1 - agent
    box_0 box_1 - box
    heavy_0 - heavybox
    up down left right - direction
    loc_1_1 loc_2_1 loc_3_1 loc_4_1 - location
    loc_1_2 loc_2_2 loc_3_2 loc_4_2 - location
    loc_1_3 loc_2_3 loc_3_3 loc_4_3 - location
    loc_1_4 loc_2_4 loc_3_4 loc_4_4 - location
  )
  (:init
    ;; --- Vertical Adjacency ---
    (adj loc_1_1 loc_1_2 down) (adj loc_1_2 loc_1_1 up)
    (adj loc_1_2 loc_1_3 down) (adj loc_1_3 loc_1_2 up)
    (adj loc_1_3 loc_1_4 down) (adj loc_1_4 loc_1_3 up)

    (adj loc_2_1 loc_2_2 down) (adj loc_2_2 loc_2_1 up)
    (adj loc_2_2 loc_2_3 down) (adj loc_2_3 loc_2_2 up)
    (adj loc_2_3 loc_2_4 down) (adj loc_2_4 loc_2_3 up)

    (adj loc_3_1 loc_3_2 down) (adj loc_3_2 loc_3_1 up)
    (adj loc_3_2 loc_3_3 down) (adj loc_3_3 loc_3_2 up)
    (adj loc_3_3 loc_3_4 down) (adj loc_3_4 loc_3_3 up)

    (adj loc_4_1 loc_4_2 down) (adj loc_4_2 loc_4_1 up)
    (adj loc_4_2 loc_4_3 down) (adj loc_4_3 loc_4_2 up)
    (adj loc_4_3 loc_4_4 down) (adj loc_4_4 loc_4_3 up)

    ;; --- Horizontal Adjacency ---
    (adj loc_1_1 loc_2_1 right) (adj loc_2_1 loc_1_1 left)
    (adj loc_2_1 loc_3_1 right) (adj loc_3_1 loc_2_1 left)
    (adj loc_3_1 loc_4_1 right) (adj loc_4_1 loc_3_1 left)

    (adj loc_1_2 loc_2_2 right) (adj loc_2_2 loc_1_2 left)
    (adj loc_2_2 loc_3_2 right) (adj loc_3_2 loc_2_2 left)
    (adj loc_3_2 loc_4_2 right) (adj loc_4_2 loc_3_2 left)

    (adj loc_1_3 loc_2_3 right) (adj loc_2_3 loc_1_3 left)
    (adj loc_2_3 loc_3_3 right) (adj loc_3_3 loc_2_3 left)
    (adj loc_3_3 loc_4_3 right) (adj loc_4_3 loc_3_3 left)

    (adj loc_1_4 loc_2_4 right) (adj loc_2_4 loc_1_4 left)
    (adj loc_2_4 loc_3_4 right) (adj loc_3_4 loc_2_4 left)
    (adj loc_3_4 loc_4_4 right) (adj loc_4_4 loc_3_4 left)

    ;; --- Entities Setup ---
    (agent-at agent_0 loc_1_1)
    (agent-at agent_1 loc_3_2)
    (box-at box_0 loc_1_2)
    (box-at box_1 loc_2_2)
    (heavybox-at heavy_0 loc_4_2)

    ;; --- Clear Tiles (Any tile that DOES NOT have a box!) ---
    (clear loc_1_1) (clear loc_2_1) (clear loc_3_1) (clear loc_4_1)
    (clear loc_3_2) ;; (1,2), (2,2), and (4,2) have boxes, so they are not clear
    (clear loc_1_3) (clear loc_2_3) (clear loc_3_3) (clear loc_4_3)
    (clear loc_1_4) (clear loc_2_4) (clear loc_3_4) (clear loc_4_4)
  )
  (:goal (and
    (box-at box_0 loc_1_4)
    (box-at box_1 loc_2_4)
    (heavybox-at heavy_0 loc_4_4)
  ))
)