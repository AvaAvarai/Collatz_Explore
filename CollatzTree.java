import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.awt.geom.AffineTransform;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

public class CollatzTree extends JPanel {
    private final HashMap<Integer, Point> positions = new HashMap<>();
    private final HashMap<Integer, Integer> collatzMap = new HashMap<>();
    private final Set<Integer> processed = new HashSet<>();
    private int current = 1;
    private int maxLevel = 1; // Tracks the longest branch
    private int longestSequence = 1; // Tracks the sequence with the longest branch
    private double scale = 1.0; // Zoom scale
    private final Timer timer;

    public CollatzTree() {
        setBackground(Color.BLACK);

        // Add mouse wheel listener for scaling
        addMouseWheelListener(new MouseWheelListener() {
            @Override
            public void mouseWheelMoved(MouseWheelEvent e) {
                int notches = e.getWheelRotation();
                if (notches < 0) {
                    // Zoom in
                    scale = Math.min(scale * 1.1, 5.0); // Max zoom-in limit
                } else {
                    // Zoom out
                    scale = Math.max(scale * 0.9, 0.001); // Max zoom-out limit
                }
                repaint();
            }
        });

        // Timer for growing the tree
        timer = new Timer(50, e -> growTree());
        timer.start();
    }

    private void growTree() {
        if (!processed.contains(current)) {
            generateCollatz(current);
        }
        current++;
        repaint();
    }

    private void generateCollatz(int n) {
        int level = 0;
        int value = n;
        while (value != 1 && !collatzMap.containsKey(value)) {
            collatzMap.put(value, nextCollatz(value));
            level++;
            value = nextCollatz(value);
        }
        processed.add(n);

        // Update the longest branch information
        if (level > maxLevel) {
            maxLevel = level;
            longestSequence = n;
        }
    }

    private int nextCollatz(int n) {
        return (n % 2 == 0) ? n / 2 : 3 * n + 1;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int width = getWidth();
        int height = getHeight();

        // Root node position: bottom-right corner
        int rootX = width - 50;
        int rootY = height - 50;

        // Scale and anchor the tree dynamically to keep the root fixed at bottom-right
        g2.translate(rootX, rootY); // Translate to root position
        g2.scale(scale, scale); // Apply the mouse-controlled scale
        g2.translate(-rootX, -rootY); // Undo the translation effect for proper layout

        positions.put(1, new Point(rootX, rootY)); // Root node

        for (int n : collatzMap.keySet()) {
            // Calculate positions dynamically
            if (!positions.containsKey(n)) {
                int parent = collatzMap.get(n);
                Point parentPos = positions.get(parent);
                if (parentPos != null) { // Ensure parent position exists
                    int level = getLevel(n);
                    int spacing = 10 + level * 2; // Reduced spread of branches
                    int x = parentPos.x + (n % 2 == 0 ? -spacing : spacing); // Horizontal spacing
                    int y = parentPos.y - 20; // Vertical spacing
                    positions.put(n, new Point(x, y));
                }
            }

            // Draw lines and nodes
            Point currentPos = positions.get(n);
            Point parentPos = positions.get(collatzMap.get(n));
            if (currentPos != null && parentPos != null) {
                // Use level to calculate hue, progressing smoothly by depth
                int level = getLevel(n);
                float hue = (float) level / maxLevel; // Normalize hue based on max depth
                g2.setColor(Color.getHSBColor(hue, 1.0f, 1.0f));
                g2.drawLine(currentPos.x, currentPos.y, parentPos.x, parentPos.y);
            }
            if (currentPos != null) {
                int level = getLevel(n);
                float hue = (float) level / maxLevel; // Normalize hue for nodes
                g2.setColor(Color.getHSBColor(hue, 1.0f, 0.8f)); // Slightly dimmer for nodes
                g2.fillOval(currentPos.x - 3, currentPos.y - 3, 6, 6); // Smaller node size
            }
        }

        // Reset scaling and translation for stats
        g2.setTransform(new AffineTransform()); // Reset all transformations

        // Display statistics
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        g2.drawString("Current Sequence Number: " + current, 10, 20);
        g2.drawString("Longest Branch: " + maxLevel, 10, 40);
        g2.drawString("Sequence with Longest Branch: " + longestSequence, 10, 60);
    }

    private int getLevel(int n) {
        int level = 0;
        while (n != 1) {
            n = nextCollatz(n);
            level++;
        }
        return level;
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Collatz Sequence Tree");
        CollatzTree tree = new CollatzTree();
        frame.add(tree);
        frame.setSize(800, 600);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}
