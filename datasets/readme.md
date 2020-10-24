# Signature Database Design
This competition consists of two separate signature verification tasks, each of which is based on a different signature database. The signature data for the first task contain coordinate information only, but the signature data for the second task also contain additional information including pen orientation and pressure. The first task is suitable for on-line signature verification on small pen-based input devices such as personal digital assistants (PDA) and the second task on digitizing tablets.

Each database has 100 sets of signature data. Each set contains 20 genuine signatures from one signature contributor and 20 skilled forgeries from five other contributors. Only 40 sets of signature data are released to participants for developing and evaluating their systems before submission. Although both genuine signatures and skilled forgeries are made available to participants, it should be noted that user enrollment during system evaluation will accept only five genuine signatures from each user, although multiple sets of five genuine signatures each will be used in multiple runs. Skilled forgeries will not be available during the enrollment process. They will only be used in the matching process for system performance evaluation. However, evaluation of signature verification performance for each user will only start after all users have finished their enrollment. Thus, participants may make use of genuine signatures from other users in improving the verification accuracy for a user. 

# Signature Files
Each signature is stored in a separate text file (with .TXT file extension). The naming convention of the signature files is UxSy, where x is the user ID and y is the signature ID. The signature files released to participants have x values ranging from 1 to 40. Genuine signatures correspond to y values from 1 to 20 and skilled forgeries from 21 to 40. 

In each signature file, the signature is simply represented as a sequence of points. The first line stores a single integer which is the total number of points in the signature. Each of the following lines corresponds to one point characterized by features listed in the following order (the last three features are missing in signature files for the first task):

* X-coordinate - scaled cursor position along the x-axis
* Y-coordinate - scaled cursor position along the y-axis
* Time stamp - system time at which the event was posted
* Button status - current button status (0 for pen-up and 1 for pen-down)
* Azimuth - clockwise rotation of cursor about the z-axis
* Altitude - angle upward toward the positive z-axis
* Pressure - adjusted state of the normal pressure

# Signature Data Collection 

Each data contributor was asked to produce 20 genuine signatures and 20 skilled forgeries in two separate sessions. For privacy reasons, the contributors were advised not to use their real signatures in daily use. Instead, they were suggested to design a new signature and to practice the writing of it sufficiently so that it remained relatively consistent over different signature instances, just like real signatures. Contributors were also reminded that consistency should not be limited to spatial consistency in the signature shape but should also include temporal consistency due to dynamic features.

In the first session, each contributor was asked to contribute 10 genuine signatures. Contributors were advised to write naturally on the digitizing tablet (WACOM Intuos tablet) as if they were enrolling themselves to a real signature verification system. They were also suggested to practice thoroughly before the actual data collection started. Moreover, contributors were provided the option of not accepting a signature instance if they were not satisfied with it. 

In the second session, which was at least one week after the first one, each contributor came again to contribute another 10 genuine signatures. In addition, he/she also contributed four skilled forgeries for each of five other contributors. Skilled forgeries were collected in the following fashion. Using a viewer, a contributor could see genuine signatures of other contributors that he/she would attempt to forge. The viewer could replay the writing sequence of the signatures on the computer screen. Contributors were also advised to practice the skilled forgeries for a few times until they were confident to proceed to the actual data collection.

The signatures are mostly in either English or Chinese. 