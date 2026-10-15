# src/rngd/hotel/layout.py
# Egress geometry: stair doors open off the corridor at mid-stair-width.
s1_door = STAIR_W_FT / 2
s2_door = length - STAIR_W_FT / 2
dead_end = max(s1_door, length - s2_door)
travel = (s2_door - s1_door) / 2 + ROOM_DEPTH_FT  # corridor run to nearest stair + walk inside the room
