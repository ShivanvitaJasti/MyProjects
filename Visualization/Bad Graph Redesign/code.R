library(tidyverse)
library(ggplot2)
library(RColorBrewer)
library(tidyr)
library(dplyr)
library(gridExtra)
library(grid)
library(usmap)
library(purrr)
library(maps)

source("C:/hw(1).R")

source("C:/hwRight(1).R")

source("C:/hwLeft(1).R")

Commute = read.csv("C:/Users/Public/Commute.csv")

Commute

Firearms = read.csv("C:/Users/Public/Firearms.csv")

Firearms

#grouped bar-chart

Commute_grouped = gather(Commute, key="Classification", value="Value", Work.from.home, Walk, Public.transport, Carpool, Drive)

Commute_grouped

Commute_grouped$Classification <- factor(Commute_grouped$Classification, levels = c('Drive','Public.transport','Carpool','Work.from.home','Walk'))

#Bad Graph - 1 : Redesign - 1 : Grouped Bar chart

plot <- ggplot(Commute_grouped) +
  aes(fill = Classification) +
  aes(x = City) +
  aes(y = Value) +
  geom_bar(stat = "identity" , width = 0.9 , position = "dodge") +
  geom_text(aes(label = round(Value, 1)), position = position_dodge(0.9), color="black",hjust = 0.5, vjust = -0.5, size = 3.5) +
  labs(title = "Modes of commutes to Work \nA study conducted in Eight different cities" ,
       x = "City",
       y = "Proportion of different modes of commute(Percentage)") +
  scale_fill_brewer(palette = "Set2") +
  theme(legend.position="bottom" , legend.title=element_blank())

plot <- plot + coord_cartesian(ylim=c(0, 80)) + 
  scale_y_continuous(breaks=seq(0, 80, 10))

plot

#Bad Graph - 1 : Redesign - 2 : Bar Chart, separated by Mode of commute

ggplot(Commute_grouped) +
  aes(fill = Classification) +
  aes(x = City) +
  aes(y = Value) +
  geom_bar(stat = "identity" , width = 0.9 , position = "dodge") +
  geom_text(aes(label = round(Value, 1)), position = position_dodge(0.9), color="black",hjust = 0.09, vjust = 0.3, size = 3.5) +
  labs(title = "Modes of commutes to Work \nA study conducted in Eight different cities" ,
       x = "City",
       y = "Proportion of different modes of commute(Percentage)") +
  scale_fill_brewer(palette = "Set2") +
  theme(legend.position="top" , legend.title=element_blank())+
  facet_wrap(~ Classification) + 
  coord_cartesian(ylim=c(0, 80)) + 
  scale_y_continuous(breaks=seq(0, 80, 10)) +
  coord_flip()

#Bad Graph - 1 : Redesign - 3 : Line Chart, separated by Mode of commute

line_walk <- ggplot(data = Commute, aes(x = City)) +
  geom_line(aes(y = Walk, colour = "Walk" , group = 1),size = 1.2) +
  geom_point(aes(y = Walk, colour = "Walk" , group = 1),size = 3) +
  geom_text(aes(y = Walk ,label = Walk),nudge_x = 0 , nudge_y = 2 , color = "black") +
  scale_colour_manual("", 
                      breaks = c("Walk"),
                      values = c("red")) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        legend.position = "none") +
  labs(title = "Modes of commutes to Work - A study conducted in Seven different cities")

  line_drive <- ggplot(data = Commute, aes(x = City)) +
   geom_line(aes(y = Drive, colour = "Drive", group = 1),size = 1.2) +
  geom_point(aes(y = Drive, colour = "Drive" , group = 1),size = 3) +
  geom_text(aes(y = Drive ,label = Drive),nudge_x = 0 , nudge_y = 10, color = "black",size = 3.5) +
    scale_colour_manual("", 
                        breaks = c("Drive"),
                        values = c("dark green")) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          legend.position = "none")
  
  line_wfh <- ggplot(data = Commute, aes(x = City)) +
    geom_line(aes(y = Work.from.home, colour = "Work.from.home", group = 1),size = 1.2) +
    geom_point(aes(y = Work.from.home, colour = "Work.from.home" , group = 1),size = 3) +
    geom_text(aes(y = Work.from.home ,label = Work.from.home),nudge_x = 0 , nudge_y = 2 , color = "black") +
    ylab("Work From\nHome") +
    scale_colour_manual("", 
                        breaks = c("Work.from.home"),
                        values = c("blue")) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          legend.position = "none")
  
  line_public <- ggplot(data = Commute, aes(x = City)) +
    geom_line(aes(y = Public.transport, colour = "Public.transport", group = 1),size = 1.2) +
    geom_point(aes(y = Public.transport, colour = "Public.transport" , group = 1),size = 3) +
    geom_text(aes(y = Public.transport ,label = Public.transport),nudge_x = 0 , nudge_y = 10, color = "black",size = 3.5) +
    ylab("Public\nTransport") +
    scale_colour_manual("", 
                        breaks = c("Public.transport"),
                        values = c("violet")) +
    theme(axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          legend.position = "none")
  
  line_carpool <- ggplot(data = Commute, aes(x = City)) +
    geom_line(aes(y = Carpool, colour = "Carpool", group = 1),size = 1.2) +
    geom_point(aes(y = Carpool, colour = "Carpool" , group = 1),size = 3) +
    geom_text(aes(y = Carpool ,label = Carpool),nudge_x = 0 , nudge_y = 2 , color = "black") +
    scale_colour_manual("", 
                        breaks = c("Carpool"),
                        values = c("orange")) +
    theme(legend.position = "none",
          axis.text.x=element_text(size = 11.5))
  
line_final <- grid.draw(rbind(ggplotGrob(line_walk), ggplotGrob(line_drive), ggplotGrob(line_wfh),ggplotGrob(line_public),  ggplotGrob(line_carpool), size = "last"))

#Bad Graph - 2 : Redesign - 1 : Grouped Scatter Plot

point1 <- ggplot(Firearms, aes(x=Background.Check, y=State)) + geom_point(colour = "black", fill = "cyan", shape = 21 ) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_text(size = 8),
        axis.title.x=element_text(size = 8)) +
  xlab("Background\nCheck") +
  scale_y_discrete(limits=rev) +
  scale_x_discrete(limits=rev)

point2 <- ggplot(Firearms, aes(x=Purchase.Permit, y=State)) + geom_point(colour = "black", fill = "lavender", shape = 21) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_text(size = 8)) +
  scale_y_discrete(limits=rev) +
  scale_x_discrete(limits=rev)

point3 <- ggplot(Firearms, aes(x=Selling.License, y=State)) + geom_point(colour = "black", fill = "yellow", shape = 21) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_text(size = 8)) +
  scale_y_discrete(limits=rev) +
  scale_x_discrete(limits=rev)

point4 <- ggplot(Firearms, aes(x=Records.Filed, y=State)) + geom_point(colour = "black", fill = "green", shape = 21) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_text(size = 8)) +
  scale_y_discrete(limits=rev) +
  scale_x_discrete(limits=rev)

point5 <- ggplot(Firearms, aes(x=Workplace.Ban, y=State)) + geom_point(colour = "black", fill = "violet", shape = 21) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_text(size = 8)) +
  scale_y_discrete(limits=rev) +
  scale_x_discrete(limits=rev)

point6 <- ggplot(Firearms, aes(x=Vote, y=State)) + geom_point(colour = "black", fill = "red", shape = 21) +
  theme(axis.text.y=element_text(size = 8),
        axis.text.x=element_text(size = 7)) +
  scale_y_discrete(limits=rev) +
  scale_x_discrete(limits=rev)

grid.arrange(point6,point1,point2,point3,point4,point5,ncol = 6,widths = c(2,2,2,2,2,2),
             top = "Conditions for Issuing firearms in the states of US") +
  theme(plot.title = element_text(face = "bold"))

#Bad Graph - 2 : Redesign - 2 : Map Chart


map1 <- plot_usmap(data= Firearms, values = "Background.Check", labels= TRUE , label_color = "white") +
  scale_fill_manual(values = c("white", "blue")) +
  ggtitle("Background Check") +
  theme(plot.title = element_text(size = 10),
        legend.position = "none")

map1$layers[[2]]$aes_params$size <- 3

map2 <- plot_usmap(data= Firearms, values = "Purchase.Permit", labels= TRUE ,  label_color = "white") +
  scale_fill_manual(values = c("white", "red")) +
  ggtitle("Permit Required to Purchase") +
  theme(plot.title = element_text(size = 10),
        legend.position = "none")

map2$layers[[2]]$aes_params$size <- 3

map3 <- plot_usmap(data= Firearms, values = "Selling.License", labels= TRUE,  label_color = "white") +
  scale_fill_manual(values = c("white", "dark green")) +
  ggtitle("License Required to sell") +
  theme(plot.title = element_text(size = 10),
        legend.position = "none")

map3$layers[[2]]$aes_params$size <- 3

map4 <- plot_usmap(data= Firearms, values = "Records.Filed", labels= TRUE,  label_color = "white") +
  scale_fill_manual(values = c("white", "brown")) +
  ggtitle("Records are filed") +
  theme(plot.title = element_text(size = 10),
        legend.position = "none")

map4$layers[[2]]$aes_params$size <- 3

map5 <- plot_usmap(data= Firearms, values = "Workplace.Ban", labels= TRUE,  label_color = "white") +
  scale_fill_manual(values = c("white", "purple")) +
  ggtitle("Firearms Banned from Workplace") +
  theme(plot.title = element_text(size = 10),
        legend.position = "none")

map5$layers[[2]]$aes_params$size <- 3

map6 <- plot_usmap(data= Firearms, values = "Vote", labels= TRUE,  label_color = "white") +
  scale_fill_manual(values = c("dark orange", "white")) +
  ggtitle("      Voted for Obama") +
  theme(plot.title = element_text(size = 10),
        legend.position = "none")

map6$layers[[2]]$aes_params$size <- 3

grid.arrange(map1, map2, map3, map4, map5, map6 ,nrow = 2, ncol = 3,
             top = grid::textGrob('Conditions for Issuing firearms in the states of US\nHighlighted States indicate Yes', gp=grid::gpar(fontsize=18))) +
  theme(plot.subtitle = element_text(size = 10))



