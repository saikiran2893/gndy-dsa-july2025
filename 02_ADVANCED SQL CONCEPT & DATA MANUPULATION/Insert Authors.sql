-- Insert Authors
INSERT INTO sample1.authors (AuthorID, AuthorName) VALUES
(1, 'J.K. Rowling'),
(2, 'George Orwell'),
(3, 'J.R.R. Tolkien'),
(4, 'Agatha Christie'),
(5, 'Stephen King');

-- Insert Books
INSERT INTO sample1.books (BookID, Title, AuthorID) VALUES
(1, 'Harry Potter and the Sorcerer''s Stone', 1),
(2, '1984', 2),
(3, 'The Hobbit', 3),
(4, 'Animal Farm', 2),
(5, 'The Lord of the Rings', 3),
(6, 'Murder on the Orient Express', 4),
(7, 'The Shining', 5),
(8, 'The Stand', 5),
(9, 'And Then There Were None', 4),
(10, 'Carrie', 5),
(11, 'The Silmarillion', 3),
(12, 'Harry Potter and the Chamber of Secrets', 1),
(13, 'Harry Potter and the Prisoner of Azkaban', 1),
(14, 'Harry Potter and the Goblet of Fire', 1),
(15, 'Harry Potter and the Order of the Phoenix', 1);

-- Insert Customers
INSERT INTO sample1.customers (CustomerID, CustomerName) VALUES
(1, 'Alice Johnson'),
(2, 'Bob Smith'),
(3, 'Charlie Brown'),
(4, 'David Williams'),
(5, 'Emily Davis'),
(6, 'Frank Harris'),
(7, 'Grace Clark'),
(8, 'Hannah Lewis'),
(9, 'Ian Walker'),
(10, 'Jack Hall'),
(11, 'Karen Allen'),
(12, 'Liam Young'),
(13, 'Mia Scott'),
(14, 'Noah Wright'),
(15, 'Olivia King'),
(16, 'Paul Green'),
(17, 'Quinn Baker'),
(18, 'Rachel Adams'),
(19, 'Samuel Nelson'),
(20, 'Taylor Carter'),
(21, 'Uma Roberts'),
(22, 'Victor Campbell'),
(23, 'Wendy Mitchell'),
(24, 'Xavier Perez'),
(25, 'Yvonne Torres');

-- Insert BorrowedBooks
INSERT INTO sample1.borrowedbooks (BorrowID, CustomerID, BookID, BorrowDate, ReturnDate) VALUES
(1, 1, 1, '2024-02-01', '2024-02-10'),
(2, 2, 3, '2024-02-05', '2024-02-15'),
(3, 3, 2, '2024-02-07', NULL),
(4, 4, 5, '2024-02-10', NULL),
(5, 5, 7, '2024-02-12', '2024-02-20'),
(6, 6, 9, '2024-02-14', NULL),
(7, 7, 11, '2024-02-16', '2024-02-25'),
(8, 8, 13, '2024-02-18', NULL),
(9, 9, 15, '2024-02-20', '2024-02-28'),
(10, 10, 4, '2024-02-22', NULL),
(11, 11, 6, '2024-02-24', '2024-03-05'),
(12, 12, 8, '2024-02-26', NULL),
(13, 13, 10, '2024-02-28', '2024-03-10'),
(14, 14, 12, '2024-03-02', NULL),
(15, 15, 14, '2024-03-04', '2024-03-15'),
(16, 16, 1, '2024-03-06', NULL),
(17, 17, 3, '2024-03-08', '2024-03-18'),
(18, 18, 5, '2024-03-10', NULL),
(19, 19, 7, '2024-03-12', '2024-03-22'),
(20, 20, 9, '2024-03-14', NULL),
(21, 21, 11, '2024-03-16', '2024-03-26'),
(22, 22, 13, '2024-03-18', NULL),
(23, 23, 15, '2024-03-20', '2024-03-30'),
(24, 24, 2, '2024-03-22', NULL),
(25, 25, 4, '2024-03-24', '2024-04-03');
