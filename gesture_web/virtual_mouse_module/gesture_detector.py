from virtual_mouse_module.utils import calculate_distance


class GestureDetector:

    def __init__(self):
        # We now use a ratio for scale invariance instead of a hardcoded distance
        self.click_ratio_threshold = 0.25

    def get_hand_size(self, hand_landmarks):
        # Calculate distance from Wrist (0) to Middle Finger MCP (9) to establish base hand size
        wrist = hand_landmarks[0]
        middle_mcp = hand_landmarks[9]
        return calculate_distance(wrist, middle_mcp)

    # Left Click
    def detect_click(self, hand_landmarks):
        hand_size = self.get_hand_size(hand_landmarks)
        if hand_size == 0: return False  # Prevent division by zero
        
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        distance = calculate_distance(thumb_tip, index_tip)

        return (distance / hand_size) < self.click_ratio_threshold

    # Right Click
    def detect_right_click(self, hand_landmarks):
        hand_size = self.get_hand_size(hand_landmarks)
        if hand_size == 0: return False
        
        thumb_tip = hand_landmarks[4]
        middle_tip = hand_landmarks[12]
        distance = calculate_distance(thumb_tip, middle_tip)

        return (distance / hand_size) < self.click_ratio_threshold

    # Drag Gesture
    def detect_drag(self, hand_landmarks):
        hand_size = self.get_hand_size(hand_landmarks)
        if hand_size == 0: return False
        
        thumb_tip = hand_landmarks[4]
        ring_tip = hand_landmarks[16]
        distance = calculate_distance(thumb_tip, ring_tip)

        return (distance / hand_size) < self.click_ratio_threshold

    # Scroll Gesture
    def detect_scroll(self, hand_landmarks):
        hand_size = self.get_hand_size(hand_landmarks)
        if hand_size == 0: return False
        
        thumb_tip = hand_landmarks[4]
        pinky_tip = hand_landmarks[20]
        distance = calculate_distance(thumb_tip, pinky_tip)

        return (distance / hand_size) < self.click_ratio_threshold