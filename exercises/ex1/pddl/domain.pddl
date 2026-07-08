(define (domain box-push)
  (:requirements :typing)
  (:types agent box heavybox location direction)
  
  (:predicates
    (agent-at ?a - agent ?l - location)
    (box-at ?b - box ?l - location)
    (heavybox-at ?h - heavybox ?l - location)
    (clear ?l - location) ;; "Clear" now strictly means "No box is here"
    (adj ?l1 - location ?l2 - location ?d - direction)
  )

  (:action move
    :parameters (?a - agent ?from - location ?to - location ?d - direction)
    :precondition (and (agent-at ?a ?from) (clear ?to) (adj ?from ?to ?d))
    :effect (and 
      (not (agent-at ?a ?from)) 
      (agent-at ?a ?to)
      ;; Agents do not remove "clear" status, so they can walk past each other!
    )
  )

  (:action push-small
    :parameters (?a - agent ?pos - location ?boxpos - location ?newboxpos - location ?d - direction ?b - box)
    :precondition (and 
      (agent-at ?a ?pos) 
      (box-at ?b ?boxpos) 
      (clear ?newboxpos)
      (adj ?pos ?boxpos ?d) 
      (adj ?boxpos ?newboxpos ?d) 
    )
    :effect (and 
      (not (agent-at ?a ?pos)) 
      (agent-at ?a ?boxpos)
      (not (box-at ?b ?boxpos)) 
      (box-at ?b ?newboxpos)
      (clear ?boxpos)         ;; The tile the box left is now clear
      (not (clear ?newboxpos)) ;; The tile the box entered is blocked
    )
  )

  (:action push-heavy
    ;; Both agents must group up at ?pos and push together!
    :parameters (?a1 - agent ?a2 - agent ?pos - location ?boxpos - location ?newboxpos - location ?d - direction ?h - heavybox)
    :precondition (and 
      (not (= ?a1 ?a2))
      (agent-at ?a1 ?pos) 
      (agent-at ?a2 ?pos) 
      (heavybox-at ?h ?boxpos) 
      (clear ?newboxpos)
      (adj ?pos ?boxpos ?d) 
      (adj ?boxpos ?newboxpos ?d) 
    )
    :effect (and 
      (not (agent-at ?a1 ?pos)) 
      (not (agent-at ?a2 ?pos)) 
      (agent-at ?a1 ?boxpos)
      (agent-at ?a2 ?boxpos)
      (not (heavybox-at ?h ?boxpos)) 
      (heavybox-at ?h ?newboxpos)
      (clear ?boxpos)
      (not (clear ?newboxpos))
    )
  )
)