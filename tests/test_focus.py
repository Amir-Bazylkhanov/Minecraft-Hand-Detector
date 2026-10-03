import unittest

from handcraft.focus import evaluate_focus


class TestFocusGuard(unittest.TestCase):
    def test_minecraft_java_accepted(self):
        self.assertTrue(evaluate_focus(
            "Minecraft 1.21.1 - Singleplayer",
            r"C:\Program Files\Java\jdk-21\bin\javaw.exe"))
        self.assertTrue(evaluate_focus("Minecraft* 1.20.4", "java.exe"))

    def test_java_process_without_minecraft_title_rejected(self):
        self.assertFalse(evaluate_focus("IntelliJ IDEA", "java.exe"))
        self.assertFalse(evaluate_focus("Some Server Console", "javaw.exe"))

    def test_minecraft_title_without_java_rejected(self):
        # e.g. a browser tab or the Bedrock edition
        self.assertFalse(evaluate_focus("Minecraft Wiki - Chrome", "chrome.exe"))
        self.assertFalse(evaluate_focus("Minecraft", "Minecraft.Windows.exe"))

    def test_fail_closed_on_missing_data(self):
        self.assertFalse(evaluate_focus(None, "javaw.exe"))
        self.assertFalse(evaluate_focus("Minecraft", None))
        self.assertFalse(evaluate_focus("", ""))
        self.assertFalse(evaluate_focus(None, None))

    def test_case_insensitive(self):
        self.assertTrue(evaluate_focus("MINECRAFT 1.8.9", "JAVAW.EXE"))


if __name__ == "__main__":
    unittest.main()
