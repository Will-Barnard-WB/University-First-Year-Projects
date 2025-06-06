import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

public class EchoClient {

    private String address;
    private int port;

    public EchoClient(String address, int port) {

        this.address = address;
        this.port = port;

    }

    private void run(){

        try(Socket socket = new Socket(address, port)){
            PrintWriter sendTo = new PrintWriter(   socket.getOutputStream(), true);

            BufferedReader readFrom = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            BufferedReader cmd = new BufferedReader(new InputStreamReader(System.in));

            System.out.print("client: ");
            String userInput = cmd.readLine();
            String response;

            while(userInput != null){

                sendTo.println(userInput);
                response = readFrom.readLine();
                System.out.println("Server: " + response);
                System.out.print("client: ");
                userInput = cmd.readLine();
            }
        }

        catch (IOException ioe) {
            ioe.printStackTrace();
        }

    }

    public static void main(String[] args){

        EchoClient client = new EchoClient("localhost",7777);
        client.run();
    }

}