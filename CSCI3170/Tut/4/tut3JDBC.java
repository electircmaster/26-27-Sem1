import java.io.Console;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.sql.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/** Lab 2: run from a terminal with mode setup, crud, prepared, budget, or workload. */
public class tut3JDBC {
    private static final String DEFAULT_URL =
        "jdbc:oracle:thin:@//db18.cse.cuhk.edu.hk:1521/oradb.cse.cuhk.edu.hk";

    public static void main(String[] args) throws Exception {
        if (args.length != 1 || !Arrays.asList("setup", "crud", "prepared", "budget", "workload").contains(args[0])) {
            throw new IllegalArgumentException("Usage: tut3JDBC setup|crud|prepared|budget|workload");
        }
        Console console = System.console();
        if (console == null) {
            throw new IllegalStateException("Run in a terminal without redirecting stdin/stdout.");
        }
        String username = console.readLine("Oracle username: ");
        char[] secret = console.readPassword("Oracle password: ");
        if (secret == null) throw new IllegalStateException("No password supplied");
        String password = new String(secret);
        Arrays.fill(secret, '\0');
        String url = System.getenv("ORACLE_JDBC_URL");
        if (url == null || url.trim().isEmpty()) url = DEFAULT_URL;
        Class.forName("oracle.jdbc.OracleDriver");
        try (Connection c = DriverManager.getConnection(url, username, password)) {
            c.setAutoCommit(false);
            try {
                if (args[0].equals("setup")) {
                    initialize(c);
                    c.commit();
                    try (PrintWriter trace = new PrintWriter(new FileWriter("jdbc_setup.txt"))) {
                        databaseSnapshot(c, trace);
                        trace.flush();
                        if (trace.checkError()) throw new java.io.IOException("Cannot write setup trace");
                    }
                    System.out.println("Setup committed; compare jdbc_setup.txt with sqlplus_setup.txt.");
                } else if (args[0].equals("prepared")) {
                    preparedLookup(c);
                    c.rollback();
                } else if (args[0].equals("crud")) {
                    try (PrintWriter trace = new PrintWriter(new FileWriter("jdbc_crud.txt"))) {
                        crud(c, trace);
                        trace.flush();
                        if (trace.checkError()) throw new java.io.IOException("Cannot write trace");
                    }
                    c.rollback();
                    System.out.println("Rolled back. Compare jdbc_crud.txt with SQL*Plus output.");
                } else {
                    if (args[0].equals("budget")) budget(c);
                    else workload(c);
                    c.commit();
                    System.out.println("COMMITTED. Verify from SQL*Plus using the same account.");
                }
            } catch (Exception e) {
                try { c.rollback(); } catch (SQLException rollbackError) {
                    e.addSuppressed(rollbackError);
                }
                if (args[0].equals("setup")) {
                    System.err.println("DDL cannot be rolled back. Reset the lab tables before retrying setup.");
                }
                throw e;
            }
        }
    }


    // Only the supplied simple SQL files: no PL/SQL, @commands, or semicolons in literals.
    static List<String> statements(String file) throws java.io.IOException {
        String text = new String(Files.readAllBytes(Paths.get("data", file)), StandardCharsets.UTF_8);
        List<String> result = new ArrayList<>();
        for (String part : text.split(";")) {
            String sql = part.trim();
            if (!sql.isEmpty()) result.add(sql);
        }
        return result;
    }

    static void initialize(Connection c) throws Exception {
        try (Statement s = c.createStatement(); ResultSet r = s.executeQuery(
                "SELECT COUNT(*) FROM USER_TABLES WHERE TABLE_NAME IN ('OFFICE','ENGINEER','PROJECT','APPLICATION','MAINTAIN','RESEARCH')")) {
            r.next();
            if (r.getInt(1) != 0) throw new SQLException("Run tut3_reset.sql first: lab tables already exist");
        }
        int created = 0;
        try (Statement s = c.createStatement()) {
            for (String sql : statements("create_table.sql")) {
                if (sql.startsWith("DROP TABLE ")) continue; // Terminal reset already did this.
                if (!sql.startsWith("CREATE TABLE ")) throw new SQLException("Unexpected DDL");
                s.executeUpdate(sql);
                created++;
            }
        }
        if (created != 6) throw new SQLException("Expected six CREATE TABLE statements");
        try (Statement s = c.createStatement()) {
            // Original office 7 address has 32 ASCII characters, exceeding VARCHAR(30).
            s.executeUpdate("ALTER TABLE OFFICE MODIFY (OADDRESS VARCHAR(50))");
        }
        System.out.println("Created 6 tables through JDBC; expanded OADDRESS to 50.");
        String[] files = {"addoffice.sql", "addengineer.sql", "addproject.sql",
                          "addapplication.sql", "addmaintain.sql", "addresearch.sql"};
        int[] expected = {8, 89, 14, 23, 26, 71};
        try (Statement s = c.createStatement()) {
            for (int i = 0; i < files.length; i++) {
                int inserted = 0;
                for (String sql : statements(files[i])) {
                    if (!sql.startsWith("INSERT INTO ")) throw new SQLException("Unexpected seed statement");
                    requireOne(s.executeUpdate(sql));
                    inserted++;
                }
                if (inserted != expected[i]) throw new SQLException("Unexpected row count in " + files[i]);
                System.out.println(files[i] + ": inserted " + inserted);
            }
        }
    }

    static void databaseSnapshot(Connection c, PrintWriter trace) throws SQLException {
        String[] queries = {
            "SELECT 'COLUMN|' || TABLE_NAME || '|' || TO_CHAR(COLUMN_ID, 'FM9990') || '|' || COLUMN_NAME || '|' || DATA_TYPE || '|' || TO_CHAR(DATA_LENGTH, 'FM9990') || '|' || NULLABLE FROM USER_TAB_COLUMNS WHERE TABLE_NAME IN ('OFFICE','ENGINEER','PROJECT','APPLICATION','MAINTAIN','RESEARCH') ORDER BY TABLE_NAME, COLUMN_ID",
            "SELECT 'KEY|' || c.TABLE_NAME || '|' || c.CONSTRAINT_TYPE || '|' || k.COLUMN_NAME || '|' || NVL(r.TABLE_NAME, '-') || '|' || NVL(r.COLUMN_NAME, '-') FROM USER_CONSTRAINTS c JOIN USER_CONS_COLUMNS k ON c.CONSTRAINT_NAME = k.CONSTRAINT_NAME LEFT JOIN USER_CONS_COLUMNS r ON c.R_CONSTRAINT_NAME = r.CONSTRAINT_NAME AND k.POSITION = r.POSITION WHERE c.TABLE_NAME IN ('OFFICE','ENGINEER','PROJECT','APPLICATION','MAINTAIN','RESEARCH') AND c.CONSTRAINT_TYPE IN ('P','R') ORDER BY c.TABLE_NAME, c.CONSTRAINT_TYPE, k.COLUMN_NAME",
            "SELECT 'DATA|OFFICE|' || TO_CHAR(OFFICEID, 'FM999999999990') || '|' || ONAME || '|' || OADDRESS || '|' || OPHONENUM FROM OFFICE ORDER BY OFFICEID",
            "SELECT 'COUNT|OFFICE|' || TO_CHAR(COUNT(*), 'FM999999990') FROM OFFICE",
            "SELECT 'DATA|ENGINEER|' || TO_CHAR(ENGINEERID, 'FM999999999990') || '|' || ENAME || '|' || TO_CHAR(ESALARY, 'FM999999999990') || '|' || TO_CHAR(OFFICEID, 'FM999999999990') FROM ENGINEER ORDER BY ENGINEERID",
            "SELECT 'COUNT|ENGINEER|' || TO_CHAR(COUNT(*), 'FM999999990') FROM ENGINEER",
            "SELECT 'DATA|PROJECT|' || TO_CHAR(PROJECTID, 'FM999999999990') || '|' || TO_CHAR(OFFICEID, 'FM999999999990') || '|' || PNAME || '|' || PCATEGORY || '|' || TO_CHAR(PBUDGET, 'FM999999999990') || '|' || TO_CHAR(PEXPENDITURE, 'FM999999999990') FROM PROJECT ORDER BY PROJECTID",
            "SELECT 'COUNT|PROJECT|' || TO_CHAR(COUNT(*), 'FM999999990') FROM PROJECT",
            "SELECT 'DATA|APPLICATION|' || TO_CHAR(APPID, 'FM999999999990') || '|' || ANAME || '|' || TO_CHAR(APRICE, 'FM999999999990') || '|' || TO_CHAR(ADATE, 'FM999999999990') || '|' || TO_CHAR(PROJECTID, 'FM999999999990') FROM APPLICATION ORDER BY APPID",
            "SELECT 'COUNT|APPLICATION|' || TO_CHAR(COUNT(*), 'FM999999990') FROM APPLICATION",
            "SELECT 'DATA|MAINTAIN|' || TO_CHAR(APPID, 'FM999999999990') || '|' || TO_CHAR(ENGINEERID, 'FM999999999990') FROM MAINTAIN ORDER BY APPID, ENGINEERID",
            "SELECT 'COUNT|MAINTAIN|' || TO_CHAR(COUNT(*), 'FM999999990') FROM MAINTAIN",
            "SELECT 'DATA|RESEARCH|' || TO_CHAR(PROJECTID, 'FM999999999990') || '|' || TO_CHAR(ENGINEERID, 'FM999999999990') FROM RESEARCH ORDER BY PROJECTID, ENGINEERID",
            "SELECT 'COUNT|RESEARCH|' || TO_CHAR(COUNT(*), 'FM999999990') FROM RESEARCH"
        };
        try (Statement s = c.createStatement()) {
            for (String sql : queries) {
                try (ResultSet r = s.executeQuery(sql)) {
                    while (r.next()) emit(trace, r.getString(1));
                }
            }
        }
    }

    static void preparedLookup(Connection c) throws SQLException {
        try (PreparedStatement p = c.prepareStatement(
                "SELECT OFFICEID, ONAME FROM OFFICE WHERE OFFICEID = ?")) {
            for (int id : new int[] {1, 7}) {
                p.setInt(1, id); // Reuse the same prepared statement with a new value.
                try (ResultSet r = p.executeQuery()) {
                    while (r.next()) System.out.println(r.getInt(1) + " | " + r.getString(2));
                }
            }
        }
    }

    static void requireOne(int count) throws SQLException {
        if (count != 1) throw new SQLException("Expected 1 affected row; got " + count);
    }

    static void emit(PrintWriter trace, String line) {
        System.out.println(line);
        trace.println(line);
    }

    // Fixed order and separators make results from the two clients comparable.
    static void snapshot(Connection c, PrintWriter trace, String stage) throws SQLException {
        emit(trace, "STAGE|" + stage);
        int count = 0;
        try (Statement s = c.createStatement(); ResultSet r = s.executeQuery(
                "SELECT OFFICEID, ONAME, OADDRESS, OPHONENUM FROM OFFICE ORDER BY OFFICEID")) {
            while (r.next()) {
                emit(trace, "ROW|" + r.getInt("OFFICEID") + "|" + r.getString("ONAME")
                    + "|" + r.getString("OADDRESS") + "|" + r.getString("OPHONENUM"));
                count++;
            }
        }
        emit(trace, "COUNT|" + count);
    }

    static void crud(Connection c, PrintWriter trace) throws SQLException {
        snapshot(c, trace, "BASELINE");
        try (PreparedStatement p = c.prepareStatement(
                "INSERT INTO OFFICE (OFFICEID, ONAME, OADDRESS, OPHONENUM) VALUES (?, ?, ?, ?)")) {
            p.setInt(1, 99);
            p.setString(2, "JDBC Demo Lab");
            p.setString(3, "Room 3170");
            p.setString(4, "20000000");
            requireOne(p.executeUpdate());
        }
        snapshot(c, trace, "INSERT");
        try (PreparedStatement p = c.prepareStatement(
                "UPDATE OFFICE SET ONAME = ?, OPHONENUM = ? WHERE OFFICEID = ?")) {
            p.setString(1, "JDBC Demo Lab Updated");
            p.setString(2, "29999999");
            p.setInt(3, 99);
            requireOne(p.executeUpdate());
        }
        snapshot(c, trace, "UPDATE");
        try (PreparedStatement p = c.prepareStatement("DELETE FROM OFFICE WHERE OFFICEID = ?")) {
            p.setInt(1, 99);
            requireOne(p.executeUpdate());
        }
        snapshot(c, trace, "DELETE");
    }

    // Integers in the supplied schema; round up to the next 100 after adding 10%.
    static int newBudget(int oldBudget, int spent) {
        if (spent <= oldBudget) return oldBudget;
        return Math.toIntExact(((spent * 110L + 9999) / 10000) * 100);
    }

    static void budget(Connection c) throws SQLException {
        List<int[]> changes = new ArrayList<>();
        // Read first. Calculate in Java, then close the ResultSet before writing.
        try (Statement s = c.createStatement(); ResultSet r = s.executeQuery(
                "SELECT PROJECTID, PNAME, PBUDGET, PEXPENDITURE FROM PROJECT ORDER BY PROJECTID")) {
            while (r.next()) {
                int id = r.getInt("PROJECTID");
                int old = r.getInt("PBUDGET");
                int spent = r.getInt("PEXPENDITURE");
                int next = newBudget(old, spent);
                if (next != old) {
                    System.out.printf("%d | %s | spent=%d | %d -> %d%n",
                        id, r.getString("PNAME"), spent, old, next);
                    changes.add(new int[] {id, old, next});
                }
            }
        }
        try (PreparedStatement p = c.prepareStatement(
                "UPDATE PROJECT SET PBUDGET = ? WHERE PROJECTID = ? AND PBUDGET = ?")) {
            for (int[] row : changes) {
                p.setInt(1, row[2]);
                p.setInt(2, row[0]);
                p.setInt(3, row[1]);
                requireOne(p.executeUpdate());
            }
        }
        System.out.println("Projects updated: " + changes.size());
    }

    static int newSalary(int oldSalary, int apps) {
        if (apps < 3) return oldSalary;
        return Math.addExact(oldSalary, 500 * Math.min(apps, 5));
    }

    static void workload(Connection c) throws SQLException {
        // Group distinct app IDs in Java instead of doing the aggregation in SQL.
        Map<Integer, Set<Integer>> appsByEngineer = new HashMap<>();
        try (Statement s = c.createStatement(); ResultSet r = s.executeQuery(
                "SELECT ENGINEERID, APPID FROM MAINTAIN")) {
            while (r.next()) {
                int id = r.getInt("ENGINEERID");
                appsByEngineer.computeIfAbsent(id, k -> new HashSet<>()).add(r.getInt("APPID"));
            }
        }
        List<int[]> changes = new ArrayList<>();
        try (Statement s = c.createStatement(); ResultSet r = s.executeQuery(
                "SELECT ENGINEERID, ENAME, ESALARY FROM ENGINEER ORDER BY ENGINEERID")) {
            while (r.next()) {
                int id = r.getInt("ENGINEERID");
                Set<Integer> apps = appsByEngineer.get(id);
                int count = apps == null ? 0 : apps.size();
                int old = r.getInt("ESALARY");
                int next = newSalary(old, count);
                if (next != old) {
                    System.out.printf("%d | %s | apps=%d | %d -> %d%n",
                        id, r.getString("ENAME"), count, old, next);
                    changes.add(new int[] {id, old, next});
                }
            }
        }
        try (PreparedStatement p = c.prepareStatement(
                "UPDATE ENGINEER SET ESALARY = ? WHERE ENGINEERID = ? AND ESALARY = ?")) {
            for (int[] row : changes) {
                p.setInt(1, row[2]);
                p.setInt(2, row[0]);
                p.setInt(3, row[1]);
                requireOne(p.executeUpdate());
            }
        }
        System.out.println("Engineers updated: " + changes.size());
    }
}
