#!/bin/bash
# Quick script to add SSH key to ssh-agent (so you don't need to enter passphrase)

echo "🔑 Adding SSH key to ssh-agent..."
echo ""

# Check if key is already added
if ssh-add -l 2>/dev/null | grep -q "id_ed25519"; then
    echo "✅ SSH key is already in ssh-agent!"
    echo ""
    echo "You can now run:"
    echo "  ./process_to_3d.sh furniture_photos/bedroom.jpg"
    exit 0
fi

# Add the key
ssh-add ~/.ssh/id_ed25519

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SSH key added to ssh-agent!"
    echo ""
    echo "Now you can run the script without entering your passphrase:"
    echo "  ./process_to_3d.sh furniture_photos/bedroom.jpg"
else
    echo ""
    echo "❌ Failed to add SSH key. Make sure you enter the correct passphrase."
fi

