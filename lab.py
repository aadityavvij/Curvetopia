phone_directory = {}

def add_contact(name, phone_numbers):
    if name in phone_directory:
        phone_directory[name].extend(phone_numbers)
        print(f"Contact {name} updated with new numbers.")
    else:
        phone_directory[name] = phone_numbers
        print(f"Contact {name} added.")

def update_contact(name, new_phone_numbers):
    if name in phone_directory:
        phone_directory[name] = new_phone_numbers
        print(f"Contact {name} updated.")
    else:
        print(f"Contact {name} does not exist.")

def delete_contact(name):
    if name in phone_directory:
        del phone_directory[name]
        print(f"Contact {name} deleted.")
    else:
        print(f"Contact {name} does not exist.")

def view_contacts():
    if phone_directory:
        for name, phone_numbers in phone_directory.items():
            print(f"Name: {name}, Phone Numbers: {', '.join(phone_numbers)}")
    else:
        print("No contacts available.")

def search_contact_by_prefix(prefix):
    found = False
    for contact_name, phone_numbers in phone_directory.items():
        if contact_name.lower().startswith(prefix.lower()):
            print(f"Name: {contact_name}, Phone Numbers: {', '.join(phone_numbers)}")
            found = True
    if not found:
        print(f"No contacts found with the prefix '{prefix}'.")

def menu():
    print("\nPhone Directory Menu:")
    print("1. Add Contact")
    print("2. Update Contact")
    print("3. Delete Contact")
    print("4. View Contacts")
    print("5. Search Contact by Prefix")
    print("6. Exit")

while True:
    menu()
    choice = input("Enter your choice: ")
    
    if choice == '1':
        name = input("Enter name: ")
        phone_numbers = input("Enter phone numbers (comma separated): ").split(',')
        add_contact(name, [num.strip() for num in phone_numbers])
    elif choice == '2':
        name = input("Enter name: ")
        new_phone_numbers = input("Enter new phone numbers (comma separated): ").split(',')
        update_contact(name, [num.strip() for num in new_phone_numbers])
    elif choice == '3':
        name = input("Enter name: ")
        delete_contact(name)
    elif choice == '4':
        view_contacts()
    elif choice == '5':
        prefix = input("Enter name prefix to search: ")
        search_contact_by_prefix(prefix)
    elif choice == '6':
        print("Exiting the phone directory.")
        break
    else:
        print("Invalid choice. Please try again.")