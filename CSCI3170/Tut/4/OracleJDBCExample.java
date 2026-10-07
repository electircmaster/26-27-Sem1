import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;


public class OracleJDBCExample {
    public static void main(String[] args) {

        // database connection informations
        String url = "jdbc:oracle:thin:@//db18.cse.cuhk.edu.hk:1521/oradb.cse.cuhk.edu.hk"; 

        // Replace with your account and password
        String username = "username"; 
        String password = "password"; 

        Connection connection = null;
        Statement statement = null;
        ResultSet resultSet = null;

        try {
            // Load JDBC Driver
            Class.forName("oracle.jdbc.driver.OracleDriver");

            // Establish a connection
            connection = DriverManager.getConnection(url, username, password);
            System.out.println("Successfully connected to the database!");

            // Create Statement Object
            statement = connection.createStatement();


            // Conduct SQL query
            String sql = "SELECT * FROM OFFICE";
            resultSet = statement.executeQuery(sql);

            // process the query result
            while (resultSet.next()) {
                String name = resultSet.getString("ONAME");
                // int id = resultSet.getInt("OFFICEID");

                // System.out.println(id);
                System.out.println(name);

            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // close the connection
            try {
                if (resultSet != null) resultSet.close();
                if (statement != null) statement.close();
                if (connection != null) connection.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    // =====================================================================================
    // Practice
    // try using JDBC to add, modify, and delete data, as well as modify column properties.
    // =====================================================================================
}